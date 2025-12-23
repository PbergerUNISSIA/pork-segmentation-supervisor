"""
====================================================================
PORK SEGMENTATION SUPERVISOR - STREAMLIT APPLICATION
====================================================================
Interface web interactive pour supervision ML avec Active Learning.
====================================================================
"""

import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image

from config.settings import settings
from core import (
    ImageProcessor,
    ModelManager,
    calculate_all_metrics,
    calculate_confidence,
    is_uncertain,
)
from database import (
    UncertaintyLevel,
    ValidationStatus,
    count_validated_annotations,
    create_annotation,
    create_image,
    create_model_version,
    create_prediction,
    get_active_model_version,
    get_dashboard_stats,
    get_image_by_checksum,
    get_images_pending_validation,
    get_recent_images,
    init_db,
)
from utils.logger import logger

st.set_page_config(
    page_title="Pork Segmentation Supervisor",
    page_icon="🐷",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def initialize_system():
    """Initialise le système au démarrage."""
    logger.info("Initializing Pork Segmentation Supervisor...")
    init_db()

    active_model = get_active_model_version()
    if not active_model:
        logger.warning("No active model found, creating default version...")
        create_model_version(
            version=settings.MODEL_VERSION,
            model_path=str(settings.MODEL_PATH),
            description="Initial U-Net INT8 quantized model",
            is_active=True
        )

    try:
        model_manager = ModelManager()
        logger.success("Model loaded successfully")
    except FileNotFoundError:
        logger.warning("Model file not found, running in demo mode")
        model_manager = None

    image_processor = ImageProcessor()
    logger.success("System initialized successfully")
    return model_manager, image_processor


model_manager, image_processor = initialize_system()


def calculate_image_checksum(image_bytes: bytes) -> str:
    """Calcule le MD5 checksum d'une image."""
    return hashlib.md5(image_bytes).hexdigest()


def save_uploaded_image(uploaded_file) -> Path:
    """Sauvegarde une image uploadée."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uploaded_file.name}"
    filepath = settings.UPLOAD_DIR / filename
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return filepath


def determine_uncertainty_level(confidence: float) -> UncertaintyLevel:
    """Détermine le niveau d'incertitude selon la confiance."""
    if confidence >= 0.75:
        return UncertaintyLevel.LOW
    elif confidence >= 0.5:
        return UncertaintyLevel.MEDIUM
    else:
        return UncertaintyLevel.HIGH


def page_home():
    """Page principale : Upload et prédiction."""
    st.title("🐷 Pork Segmentation Supervisor")
    st.markdown("---")

    st.markdown("""
    ### 📤 Upload une image de carcasse

    Le système va automatiquement :
    1. Charger le modèle TFLite INT8
    2. Prédire le masque de segmentation
    3. Calculer la confiance (entropie)
    4. Proposer validation ou correction
    """)

    uploaded_file = st.file_uploader(
        "Choisir une image",
        type=settings.get_supported_formats_list(),
        help=f"Formats supportés : {settings.SUPPORTED_FORMATS}"
    )

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Image Originale")
            image_pil = Image.open(uploaded_file)
            st.image(image_pil, use_column_width=True)
            st.caption(f"**Nom** : {uploaded_file.name}")
            st.caption(f"**Taille** : {image_pil.size[0]} x {image_pil.size[1]} px")
            st.caption(f"**Format** : {image_pil.format}")

        if st.button("🔮 Lancer la Prédiction", type="primary"):
            process_prediction(uploaded_file, image_pil, col2)


def process_prediction(uploaded_file, image_pil, display_col):
    """Traite la prédiction d'une image uploadée."""
    if model_manager is None:
        st.error("❌ Modèle TFLite introuvable. Placez votre modèle dans `data/models/`")
        return

    with st.spinner("🔄 Prédiction en cours..."):
        start_time = time.time()

        image_bytes = uploaded_file.getvalue()
        checksum = calculate_image_checksum(image_bytes)
        existing_image = get_image_by_checksum(checksum)

        if existing_image:
            st.warning(f"⚠️ Image déjà traitée : {existing_image.filename}")
            return

        filepath = save_uploaded_image(uploaded_file)

        image_record = create_image(
            filename=uploaded_file.name,
            filepath=str(filepath),
            width=image_pil.size[0],
            height=image_pil.size[1],
            file_size=len(image_bytes),
            checksum=checksum
        )

        image_array = np.array(image_pil)
        model_size = model_manager.input_shape[1]
        resized = image_processor.resize(image_array, target_size=(model_size, model_size))
        normalized = image_processor.normalize(resized, mode="float")

        probabilities = model_manager.predict(normalized, return_probabilities=True)
        mask = model_manager.predict(normalized, return_probabilities=False)

        inference_time = (time.time() - start_time) * 1000

        confidence = calculate_confidence(probabilities, method=settings.UNCERTAINTY_METHOD)
        uncertain = is_uncertain(probabilities)
        uncertainty_level = determine_uncertainty_level(confidence)

        mask_filename = f"mask_{image_record.id}.png"
        mask_path = settings.UPLOAD_DIR / mask_filename
        image_processor.save_mask(mask, mask_path)

        active_model = get_active_model_version()
        prediction_record = create_prediction(
            image_id=image_record.id,
            model_version_id=active_model.id,
            mask_path=str(mask_path),
            confidence_score=confidence,
            uncertainty_level=uncertainty_level,
            inference_time_ms=inference_time
        )

        with display_col:
            st.subheader("🎯 Résultat de Prédiction")

            overlay = image_processor.create_overlay(resized, mask, color=(255, 0, 0), alpha=0.5)
            st.image(overlay, use_column_width=True, caption="Masque prédit (rouge)")

            st.markdown("### 📊 Métriques")
            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.metric(
                    "Confiance",
                    f"{confidence:.1%}",
                    delta="Haute" if confidence >= 0.75 else "Faible",
                    delta_color="normal" if confidence >= 0.75 else "inverse"
                )

            with metric_col2:
                st.metric("Temps Inférence", f"{inference_time:.0f} ms")

            with metric_col3:
                st.metric("Incertitude", uncertainty_level.value.upper())

            if uncertain:
                st.warning("⚠️ **Confiance faible** : Validation humaine recommandée")
            else:
                st.success("✅ **Confiance élevée** : Prédiction fiable")

            st.markdown("### ✅ Actions")
            action_col1, action_col2 = st.columns(2)

            with action_col1:
                if st.button("✅ Valider"):
                    validate_prediction(image_record.id, str(mask_path), was_corrected=False)
                    st.success("✅ Annotation validée !")
                    st.rerun()

            with action_col2:
                if st.button("❌ Rejeter"):
                    st.info("Image rejetée (non implémenté dans MVP)")


def validate_prediction(image_id: int, mask_path: str, was_corrected: bool):
    """Valide une prédiction et crée une annotation."""
    create_annotation(
        image_id=image_id,
        mask_path=mask_path,
        validation_status=ValidationStatus.VALIDATED,
        was_corrected=was_corrected
    )

    validated_count = count_validated_annotations()
    if validated_count >= settings.RETRAINING_THRESHOLD:
        st.balloons()
        st.success(f"🎉 {validated_count} annotations validées ! Re-training possible.")


def page_dashboard():
    """Dashboard de statistiques et métriques."""
    st.title("📊 Dashboard de Supervision")
    st.markdown("---")

    stats = get_dashboard_stats()

    st.markdown("### 📈 Indicateurs Clés")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.metric("Total Images", stats['total_images'])
    with kpi_col2:
        st.metric("Prédictions", stats['total_predictions'])
    with kpi_col3:
        st.metric("Annotations Validées", stats['validated_annotations'])
    with kpi_col4:
        st.metric("En Attente", stats['pending_validation'])

    st.markdown("---")
    st.markdown("### 🎯 Performance (7 derniers jours)")

    perf_col1, perf_col2, perf_col3 = st.columns(3)
    with perf_col1:
        st.metric("Confiance Moyenne", f"{stats['avg_confidence_7d']:.1%}")
    with perf_col2:
        st.metric("Dice Score Moyen", f"{stats['avg_dice_7d']:.3f}" if stats['avg_dice_7d'] > 0 else "N/A")
    with perf_col3:
        st.metric("Modèle Actif", stats['active_model'] or "N/A")

    st.markdown("---")
    st.markdown("### 🔄 Progression Re-training")

    validated = stats['validated_annotations']
    threshold = settings.RETRAINING_THRESHOLD
    progress = min(validated / threshold, 1.0)

    st.progress(progress)
    st.caption(f"**{validated} / {threshold}** annotations validées ({threshold - validated} restantes)")

    if validated >= threshold:
        st.success("🎉 Seuil atteint ! Re-training possible.")


def page_validation():
    """Page de validation des images en attente."""
    st.title("📋 Validation des Prédictions")
    st.markdown("---")

    pending_images = get_images_pending_validation(limit=20)

    if not pending_images:
        st.success("✅ Aucune image en attente de validation !")
        return

    st.info(f"📝 **{len(pending_images)}** images en attente de validation")

    for img_record in pending_images:
        with st.container():
            st.markdown(f"### 🖼️ {img_record.filename}")
            col1, col2 = st.columns(2)

            try:
                image = image_processor.load_image(img_record.filepath)
                with col1:
                    st.image(image, caption="Image originale", use_column_width=True)

                if img_record.predictions:
                    latest_pred = img_record.predictions[-1]
                    mask = image_processor.load_image(latest_pred.mask_path)
                    overlay = image_processor.create_overlay(image, mask)

                    with col2:
                        st.image(overlay, caption="Prédiction", use_column_width=True)

                    st.caption(f"Confiance : **{latest_pred.confidence_score:.1%}**")
                    st.caption(f"Incertitude : **{latest_pred.uncertainty_level.value}**")

                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if st.button(f"✅ Valider", key=f"validate_{img_record.id}"):
                            validate_prediction(img_record.id, latest_pred.mask_path, False)
                            st.success("Validé !")
                            st.rerun()
                    with btn_col2:
                        if st.button(f"❌ Rejeter", key=f"reject_{img_record.id}"):
                            st.info("Rejeté")
            except Exception as e:
                st.error(f"Erreur : {e}")

            st.markdown("---")


def page_history():
    """Page d'historique des images traitées."""
    st.title("📜 Historique des Images")
    st.markdown("---")

    recent_images = get_recent_images(limit=50)

    if not recent_images:
        st.info("Aucune image traitée pour le moment.")
        return

    st.info(f"📚 **{len(recent_images)}** images récentes")

    data = []
    for img in recent_images:
        has_annotation = img.annotation is not None
        has_prediction = len(img.predictions) > 0
        confidence = img.predictions[-1].confidence_score if has_prediction else None

        data.append({
            "ID": img.id,
            "Nom": img.filename,
            "Upload": img.uploaded_at.strftime("%Y-%m-%d %H:%M"),
            "Prédictions": len(img.predictions),
            "Confiance": f"{confidence:.1%}" if confidence else "N/A",
            "Validé": "✅" if has_annotation else "❌"
        })

    st.table(data)


def page_active_learning():
    """Page Active Learning et monitoring."""
    st.title("🎯 Active Learning & Monitoring")
    st.markdown("---")

    # === 1. ACTIVE LEARNING SELECTION ===
    st.header("1️⃣ Sélection Intelligente d'Échantillons")

    from core import ActiveLearningEngine, SamplingStrategy

    col1, col2 = st.columns(2)

    with col1:
        strategy = st.selectbox(
            "Stratégie de sélection",
            [
                ("Uncertainty Entropy", SamplingStrategy.UNCERTAINTY_ENTROPY),
                ("Uncertainty Variance", SamplingStrategy.UNCERTAINTY_VARIANCE),
                ("Hybrid (Recommandé)", SamplingStrategy.HYBRID_UNCERTAINTY_DIVERSITY),
                ("Diversity K-Means", SamplingStrategy.DIVERSITY_KMEANS),
                ("Core-set", SamplingStrategy.REPRESENTATIVE_CORESET),
            ],
            format_func=lambda x: x[0]
        )

    with col2:
        n_samples = st.slider("Nombre d'échantillons", 5, 50, 10)

    if st.button("🎯 Sélectionner Échantillons", type="primary"):
        with st.spinner("Sélection en cours..."):
            # Récupérer les prédictions non validées
            from database import get_session, Prediction, Image, Annotation

            with get_session() as session:
                # Images avec prédictions mais sans annotation
                predictions_query = (
                    session.query(Prediction)
                    .join(Image)
                    .outerjoin(Annotation)
                    .filter(Annotation.id == None)
                    .all()
                )

                if not predictions_query:
                    st.warning("Aucune prédiction non validée disponible")
                else:
                    # Convertir en format dict
                    pred_dicts = [{
                        'id': p.id,
                        'image_id': p.image_id,
                        'confidence_score': p.confidence_score or 0.5,
                        'entropy': p.entropy or 0.0,
                        'variance': p.variance or 0.0,
                        'margin': p.margin or 0.5,
                        'uncertainty_level': p.uncertainty_level.value
                    } for p in predictions_query]

                    # Sélectionner candidats
                    engine = ActiveLearningEngine(
                        strategy=strategy[1],
                        diversity_weight=0.6
                    )

                    candidates = engine.select_samples_for_annotation(
                        pred_dicts,
                        n_samples=min(n_samples, len(pred_dicts))
                    )

                    st.success(f"✅ {len(candidates)} images sélectionnées")

                    # Afficher les candidats
                    for i, candidate in enumerate(candidates[:10], 1):
                        with st.expander(f"#{i} - Image ID {candidate.image_id} - Score: {candidate.score:.3f}"):
                            col_a, col_b, col_c = st.columns(3)

                            with col_a:
                                st.metric("Score d'importance", f"{candidate.score:.3f}")
                            with col_b:
                                st.metric("Confiance", f"{candidate.confidence:.1%}")
                            with col_c:
                                st.metric("Incertitude", candidate.uncertainty_level)

    st.markdown("---")

    # === 2. TRAINING PIPELINE ===
    st.header("2️⃣ Pipeline de Réentraînement")

    from core import TrainingPipeline

    pipeline = TrainingPipeline()
    stats = pipeline.get_training_stats()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Annotations Validées",
            stats['validated_annotations']
        )

    with col2:
        st.metric(
            "Seuil Re-training",
            stats['retraining_threshold']
        )

    with col3:
        ready = "✅ PRÊT" if stats['ready_for_training'] else "⏳ EN ATTENTE"
        st.metric("Status", ready)

    # Progress bar
    progress = min(stats['validated_annotations'] / stats['retraining_threshold'], 1.0)
    st.progress(progress)

    remaining = max(0, stats['retraining_threshold'] - stats['validated_annotations'])
    if remaining > 0:
        st.caption(f"Encore {remaining} annotations nécessaires")

    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if st.button("🚀 Lancer Re-training", disabled=not stats['ready_for_training'], type="primary"):
            with st.spinner("⏳ Re-training en cours (cela peut prendre du temps)..."):
                result = pipeline.run_training_pipeline()

                if result['success']:
                    st.success("✅ Nouveau modèle créé!")
                    if result.get('deployed'):
                        st.balloons()
                        st.success(f"🎉 Modèle {result['new_version']} déployé!")
                    else:
                        st.info(f"Modèle créé mais non déployé (amélioration insuffisante)")
                else:
                    st.error(f"❌ Erreur: {result.get('message', 'Unknown error')}")

    with col_btn2:
        if st.button("📊 Voir Stats Pipeline"):
            st.json(stats)

    st.markdown("---")

    # === 3. DRIFT DETECTION ===
    st.header("3️⃣ Détection de Drift")

    from core import DriftDetector

    detector = DriftDetector()

    col1, col2 = st.columns(2)

    with col1:
        ref_days = st.number_input("Période référence (jours)", 7, 90, 30)

    with col2:
        cur_days = st.number_input("Période actuelle (jours)", 1, 30, 7)

    if st.button("🔍 Détecter Drift", type="primary"):
        with st.spinner("Analyse en cours..."):
            try:
                drift_report = detector.detect_drift(
                    reference_period_days=ref_days,
                    current_period_days=cur_days
                )

                if drift_report.get('drift_detected'):
                    st.error(f"⚠️ DRIFT DÉTECTÉ: {drift_report.get('drift_type', 'unknown')}")
                    st.warning(f"Sévérité: {drift_report.get('severity', 'unknown')}")

                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("PSI Score", f"{drift_report.get('avg_psi', 0):.3f}")
                    with col_b:
                        st.metric("Échantillons Ref.", drift_report.get('reference_samples', 0))
                    with col_c:
                        st.metric("Échantillons Actuels", drift_report.get('current_samples', 0))

                    st.subheader("📋 Recommandations")
                    for rec in drift_report.get('recommendations', []):
                        st.info(f"• {rec}")
                else:
                    st.success("✅ Pas de drift détecté - Modèle stable")

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("PSI Score", f"{drift_report.get('avg_psi', 0):.3f}")
                    with col_b:
                        st.metric("Stabilité", "Excellente")

            except Exception as e:
                st.error(f"Erreur lors de la détection: {str(e)}")

    st.markdown("---")

    # === 4. MODEL VERSIONING ===
    st.header("4️⃣ Versions de Modèles")

    from core import ModelVersioningManager
    import pandas as pd

    manager = ModelVersioningManager()
    history = manager.get_version_history(limit=10)

    if history:
        df = pd.DataFrame(history)
        df['Status'] = df['is_active'].apply(lambda x: "🟢 ACTIF" if x else "⚪ Inactif")
        df['Created'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M')

        st.dataframe(
            df[['Status', 'version', 'metrics_dice', 'metrics_iou', 'Created', 'description']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Aucun historique de version disponible")


def main():
    """Point d'entrée principal de l'application."""
    with st.sidebar:
        st.title("🐷 Navigation")
        page = st.radio(
            "Choisir une page",
            ["🏠 Home", "📋 Validation", "📊 Dashboard", "📜 Historique", "🎯 Active Learning"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### ⚙️ Configuration")
        st.caption(f"**Environnement** : {settings.ENVIRONMENT}")
        st.caption(f"**Modèle** : {settings.MODEL_VERSION}")
        st.caption(f"**Seuil Confiance** : {settings.MODEL_CONFIDENCE_THRESHOLD:.0%}")
        st.caption(f"**Seuil Re-training** : {settings.RETRAINING_THRESHOLD}")

        st.markdown("---")
        stats = get_dashboard_stats()
        st.markdown("### 📊 Stats Rapides")
        st.metric("Images", stats['total_images'])
        st.metric("Validées", stats['validated_annotations'])

    if page == "🏠 Home":
        page_home()
    elif page == "📋 Validation":
        page_validation()
    elif page == "📊 Dashboard":
        page_dashboard()
    elif page == "📜 Historique":
        page_history()
    elif page == "🎯 Active Learning":
        page_active_learning()


if __name__ == "__main__":
    main()
