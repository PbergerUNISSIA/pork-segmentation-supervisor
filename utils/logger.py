"""
====================================================================
PORK SEGMENTATION SUPERVISOR - LOGGING SYSTEM
====================================================================
Système de logging avancé avec Loguru.

FONCTIONNALITÉS :
- Logs colorés dans le terminal
- Rotation automatique des fichiers
- Rétention temporelle
- Niveaux de log configurables
- Contexte enrichi (fonction, ligne, module)

USAGE :
    from utils.logger import logger

    logger.info("Application démarrée")
    logger.debug("Chargement du modèle...")
    logger.warning("Confiance faible détectée")
    logger.error("Erreur lors de l'inférence")
    logger.critical("Modèle introuvable !")

CONTEXTE AUTOMATIQUE :
    Les logs incluent automatiquement :
    - Timestamp
    - Niveau (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - Module source
    - Fonction source
    - Numéro de ligne
====================================================================
"""

import sys
from pathlib import Path

from loguru import logger

from config.settings import settings


def setup_logger() -> None:
    """
    Configure le système de logging avec Loguru.

    CONFIGURATION :
    - Console : Logs colorés avec format enrichi
    - Fichier : Logs persistés avec rotation et rétention
    - Niveau : Configurable via .env (LOG_LEVEL)

    ROTATION :
    - Par taille : "10 MB" (default)
    - Par temps : "1 day", "1 week"

    RÉTENTION :
    - "30 days" : Garde les logs 30 jours
    - "1 week" : Garde les logs 1 semaine
    """

    # ==============================================================
    # SUPPRESSION DES HANDLERS PAR DÉFAUT
    # ==============================================================
    # Loguru a un handler console par défaut, on le retire
    # pour le remplacer par notre configuration personnalisée
    logger.remove()

    # ==============================================================
    # CONSOLE HANDLER (Terminal coloré)
    # ==============================================================
    logger.add(
        sink=sys.stderr,  # Sortie vers stderr (standard pour logs)
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level=settings.LOG_LEVEL,  # Niveau depuis .env
        colorize=True,  # Couleurs activées
        backtrace=True,  # Afficher la stack trace complète
        diagnose=True  # Afficher les variables lors des erreurs
    )

    # ==============================================================
    # FILE HANDLER (Fichier persisté)
    # ==============================================================
    # Créer le répertoire de logs si inexistant
    log_file = Path(settings.LOG_FILE)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        sink=str(log_file),  # Chemin du fichier
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
        level=settings.LOG_LEVEL,
        rotation=settings.LOG_ROTATION,  # Ex: "10 MB" ou "1 day"
        retention=settings.LOG_RETENTION,  # Ex: "30 days"
        compression="zip",  # Compression des vieux logs
        backtrace=True,
        diagnose=True,
        encoding="utf-8"
    )

    # ==============================================================
    # LOG INITIAL
    # ==============================================================
    logger.info("=" * 60)
    logger.info("PORK SEGMENTATION SUPERVISOR - LOGGING INITIALIZED")
    logger.info("=" * 60)
    logger.info(f"Log Level: {settings.LOG_LEVEL}")
    logger.info(f"Log File: {log_file}")
    logger.info(f"Rotation: {settings.LOG_ROTATION}")
    logger.info(f"Retention: {settings.LOG_RETENTION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info("=" * 60)


# ==================================================================
# DECORATORS UTILITAIRES
# ==================================================================

def log_function_call(func):
    """
    Décorateur pour logger automatiquement les appels de fonction.

    USAGE :
        @log_function_call
        def process_image(image_path: str) -> np.ndarray:
            ...

    RÉSULTAT :
        INFO | Calling process_image with args=('path/to/image.jpg',)
        INFO | process_image completed in 0.45s
    """
    from functools import wraps
    import time

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Log avant l'appel
        logger.debug(
            f"Calling {func.__name__} with args={args[:2]}... kwargs={list(kwargs.keys())}"
        )

        # Mesurer le temps d'exécution
        start_time = time.time()

        try:
            # Exécuter la fonction
            result = func(*args, **kwargs)

            # Log après succès
            elapsed = time.time() - start_time
            logger.debug(f"{func.__name__} completed in {elapsed:.2f}s")

            return result

        except Exception as e:
            # Log en cas d'erreur
            elapsed = time.time() - start_time
            logger.error(
                f"{func.__name__} failed after {elapsed:.2f}s with error: {e}"
            )
            raise

    return wrapper


# ==================================================================
# CONTEXT MANAGER POUR LOGS TEMPORAIRES
# ==================================================================

class LogContext:
    """
    Context manager pour ajouter du contexte temporaire aux logs.

    USAGE :
        with LogContext(image_id=123, user="operator_1"):
            logger.info("Processing image")
            # Log output: ... | image_id=123 user=operator_1 | Processing image
    """

    def __init__(self, **context):
        """
        Args:
            **context: Paires clé-valeur de contexte
        """
        self.context = context
        self.token = None

    def __enter__(self):
        # Ajouter le contexte
        self.token = logger.contextualize(**self.context)
        self.token.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Retirer le contexte
        if self.token:
            self.token.__exit__(exc_type, exc_val, exc_tb)


# ==================================================================
# INITIALISATION AUTOMATIQUE
# ==================================================================

# Configurer le logger dès l'import du module
setup_logger()


# ==================================================================
# USAGE EXAMPLES
# ==================================================================

if __name__ == "__main__":
    """
    Exemples d'utilisation du logger.
    """

    # Logs basiques
    logger.debug("Ceci est un message DEBUG (développement)")
    logger.info("Ceci est un message INFO (informations générales)")
    logger.success("Ceci est un message SUCCESS (opération réussie)")
    logger.warning("Ceci est un WARNING (attention)")
    logger.error("Ceci est une ERROR (erreur récupérable)")
    logger.critical("Ceci est un message CRITICAL (erreur fatale)")

    # Log avec contexte
    with LogContext(image_id=42, model_version="v1.0"):
        logger.info("Traitement de l'image avec contexte")

    # Log avec variables
    image_path = "/path/to/image.jpg"
    confidence = 0.89
    logger.info(f"Prédiction pour {image_path} avec confiance {confidence:.2%}")

    # Décorateur de fonction
    @log_function_call
    def example_function(x: int, y: int) -> int:
        """Fonction exemple pour tester le décorateur."""
        return x + y

    result = example_function(10, 20)
    logger.info(f"Résultat: {result}")

    # Exception logging
    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception("Erreur de division par zéro détectée")

    logger.info("Tests de logging terminés !")
