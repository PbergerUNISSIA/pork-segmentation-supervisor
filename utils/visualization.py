"""
====================================================================
PORK SEGMENTATION SUPERVISOR - VISUALIZATION HELPERS
====================================================================
Fonctions utilitaires pour visualisation dans Streamlit.

FONCTIONNALITÉS :
- Création de graphiques Plotly
- Heatmaps de confiance
- Comparaisons avant/après
- Métriques formatées pour UI

USAGE :
    from utils.visualization import (
        plot_confidence_distribution,
        plot_metrics_over_time,
        format_metric_card
    )

====================================================================
"""

from typing import List, Optional

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ==================================================================
# PLOTLY CHARTS
# ==================================================================

def plot_confidence_distribution(
    confidence_scores: List[float],
    title: str = "Distribution de Confiance"
) -> go.Figure:
    """
    Crée un histogramme de la distribution des scores de confiance.

    Args:
        confidence_scores: Liste de scores [0-1]
        title: Titre du graphique

    Returns:
        go.Figure: Graphique Plotly
    """
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=confidence_scores,
        nbinsx=20,
        name="Confiance",
        marker_color='rgb(55, 83, 109)'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Score de Confiance",
        yaxis_title="Nombre de Prédictions",
        showlegend=False,
        height=400
    )

    # Ajouter lignes de seuil
    fig.add_vline(
        x=0.75,
        line_dash="dash",
        line_color="green",
        annotation_text="Seuil Haute Confiance"
    )

    fig.add_vline(
        x=0.5,
        line_dash="dash",
        line_color="orange",
        annotation_text="Seuil Incertitude"
    )

    return fig


def plot_metrics_over_time(
    dates: List,
    dice_scores: List[float],
    iou_scores: List[float],
    title: str = "Évolution des Métriques"
) -> go.Figure:
    """
    Crée un graphique d'évolution temporelle des métriques.

    Args:
        dates: Liste de dates
        dice_scores: Scores Dice
        iou_scores: Scores IoU
        title: Titre du graphique

    Returns:
        go.Figure: Graphique Plotly
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=dice_scores,
        mode='lines+markers',
        name='Dice Score',
        line=dict(color='rgb(55, 83, 109)', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=dates,
        y=iou_scores,
        mode='lines+markers',
        name='IoU Score',
        line=dict(color='rgb(26, 118, 255)', width=2)
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Score",
        yaxis_range=[0, 1],
        hovermode='x unified',
        height=400
    )

    return fig


def plot_uncertainty_pie(
    low_count: int,
    medium_count: int,
    high_count: int,
    title: str = "Répartition par Niveau d'Incertitude"
) -> go.Figure:
    """
    Crée un graphique camembert des niveaux d'incertitude.

    Args:
        low_count: Nombre de prédictions LOW
        medium_count: Nombre de prédictions MEDIUM
        high_count: Nombre de prédictions HIGH
        title: Titre du graphique

    Returns:
        go.Figure: Graphique Plotly
    """
    labels = ['Faible', 'Moyenne', 'Haute']
    values = [low_count, medium_count, high_count]
    colors = ['#00CC96', '#FFA15A', '#EF553B']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hole=0.3
    )])

    fig.update_layout(
        title=title,
        height=400
    )

    return fig


def plot_validation_progress(
    validated_count: int,
    threshold: int,
    title: str = "Progression vers Re-training"
) -> go.Figure:
    """
    Crée une barre de progression vers le re-training.

    Args:
        validated_count: Nombre d'annotations validées
        threshold: Seuil pour re-training
        title: Titre du graphique

    Returns:
        go.Figure: Graphique Plotly
    """
    progress_pct = min(validated_count / threshold * 100, 100)
    remaining_pct = 100 - progress_pct

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=['Progress'],
        x=[progress_pct],
        name='Validées',
        orientation='h',
        marker=dict(color='rgb(55, 83, 109)'),
        text=f"{validated_count} / {threshold}",
        textposition='inside'
    ))

    fig.add_trace(go.Bar(
        y=['Progress'],
        x=[remaining_pct],
        name='Restantes',
        orientation='h',
        marker=dict(color='rgb(200, 200, 200)'),
        text=f"{threshold - validated_count}",
        textposition='inside'
    ))

    fig.update_layout(
        title=title,
        barmode='stack',
        showlegend=True,
        height=150,
        xaxis=dict(range=[0, 100], showticklabels=False),
        yaxis=dict(showticklabels=False)
    )

    return fig


def plot_inference_time_distribution(
    inference_times: List[float],
    title: str = "Distribution des Temps d'Inférence"
) -> go.Figure:
    """
    Crée un graphique de distribution des temps d'inférence.

    Args:
        inference_times: Liste de temps en ms
        title: Titre du graphique

    Returns:
        go.Figure: Graphique Plotly
    """
    fig = go.Figure()

    fig.add_trace(go.Box(
        y=inference_times,
        name="Temps d'inférence",
        marker_color='rgb(26, 118, 255)',
        boxmean='sd'  # Afficher moyenne et std
    ))

    fig.update_layout(
        title=title,
        yaxis_title="Temps (ms)",
        showlegend=False,
        height=400
    )

    return fig


# ==================================================================
# HEATMAPS
# ==================================================================

def create_uncertainty_heatmap(
    probabilities: np.ndarray,
    title: str = "Carte d'Incertitude"
) -> go.Figure:
    """
    Crée une heatmap de l'incertitude pixel par pixel.

    Args:
        probabilities: Probabilités (H, W, num_classes)
        title: Titre du graphique

    Returns:
        go.Figure: Heatmap Plotly
    """
    # Calculer entropie par pixel
    epsilon = 1e-10
    probs_clipped = np.clip(probabilities, epsilon, 1.0)
    pixel_entropy = -np.sum(probs_clipped * np.log(probs_clipped), axis=-1)

    # Normaliser
    max_entropy = np.log(probabilities.shape[-1])
    normalized_entropy = pixel_entropy / max_entropy

    fig = go.Figure(data=go.Heatmap(
        z=normalized_entropy,
        colorscale='RdYlGn_r',  # Rouge = incertain, Vert = certain
        colorbar=dict(title="Incertitude")
    ))

    fig.update_layout(
        title=title,
        height=500
    )

    return fig


# ==================================================================
# FORMATTING HELPERS
# ==================================================================

def format_metric_card(
    metric_name: str,
    metric_value: float,
    format_type: str = "percentage"
) -> str:
    """
    Formate une métrique pour affichage en card.

    Args:
        metric_name: Nom de la métrique
        metric_value: Valeur
        format_type: Type de formatage ("percentage", "decimal", "integer")

    Returns:
        str: Métrique formatée
    """
    if format_type == "percentage":
        return f"**{metric_name}** : {metric_value:.1%}"
    elif format_type == "decimal":
        return f"**{metric_name}** : {metric_value:.3f}"
    elif format_type == "integer":
        return f"**{metric_name}** : {int(metric_value)}"
    else:
        return f"**{metric_name}** : {metric_value}"


def format_confidence_badge(confidence: float) -> str:
    """
    Crée un badge HTML coloré pour la confiance.

    Args:
        confidence: Score de confiance [0-1]

    Returns:
        str: HTML du badge
    """
    if confidence >= 0.75:
        color = "green"
        label = "HAUTE"
    elif confidence >= 0.5:
        color = "orange"
        label = "MOYENNE"
    else:
        color = "red"
        label = "FAIBLE"

    return f"""
    <div style="
        display: inline-block;
        padding: 5px 10px;
        background-color: {color};
        color: white;
        border-radius: 5px;
        font-weight: bold;
    ">
        {label} ({confidence:.1%})
    </div>
    """


def format_status_badge(status: str) -> str:
    """
    Crée un badge HTML pour le statut de validation.

    Args:
        status: Statut (VALIDATED, PENDING, REJECTED, etc.)

    Returns:
        str: HTML du badge
    """
    colors = {
        "VALIDATED": "green",
        "PENDING": "orange",
        "REJECTED": "red",
        "AUTO_APPROVED": "blue"
    }

    color = colors.get(status.upper(), "gray")

    return f"""
    <div style="
        display: inline-block;
        padding: 5px 10px;
        background-color: {color};
        color: white;
        border-radius: 5px;
        font-weight: bold;
    ">
        {status.upper()}
    </div>
    """


# ==================================================================
# COMPARISON HELPERS
# ==================================================================

def create_comparison_plot(
    image1: np.ndarray,
    image2: np.ndarray,
    title1: str = "Avant",
    title2: str = "Après",
    main_title: str = "Comparaison"
) -> go.Figure:
    """
    Crée un graphique de comparaison côte-à-côte.

    Args:
        image1: Première image
        image2: Deuxième image
        title1: Titre image 1
        title2: Titre image 2
        main_title: Titre principal

    Returns:
        go.Figure: Graphique Plotly
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(title1, title2)
    )

    # Image 1
    fig.add_trace(
        go.Image(z=image1),
        row=1, col=1
    )

    # Image 2
    fig.add_trace(
        go.Image(z=image2),
        row=1, col=2
    )

    fig.update_layout(
        title=main_title,
        height=500,
        showlegend=False
    )

    return fig


# ==================================================================
# USAGE EXAMPLE
# ==================================================================

if __name__ == "__main__":
    """
    Test des fonctions de visualisation.
    """
    print("=" * 60)
    print("TESTING VISUALIZATION HELPERS")
    print("=" * 60)

    # Test confidence distribution
    import random
    confidence_scores = [random.random() for _ in range(100)]
    fig = plot_confidence_distribution(confidence_scores)
    print("✅ Confidence distribution plot created")

    # Test uncertainty pie
    fig = plot_uncertainty_pie(60, 30, 10)
    print("✅ Uncertainty pie chart created")

    # Test validation progress
    fig = plot_validation_progress(35, 50)
    print("✅ Validation progress bar created")

    # Test metric formatting
    metric_str = format_metric_card("Dice Score", 0.89, "percentage")
    print(f"✅ Formatted metric: {metric_str}")

    # Test badges
    badge = format_confidence_badge(0.92)
    print(f"✅ Confidence badge created")

    print("\n" + "=" * 60)
    print("✅ VISUALIZATION HELPERS TEST PASSED")
    print("=" * 60)
