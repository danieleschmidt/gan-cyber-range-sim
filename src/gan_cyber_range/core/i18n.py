"""
Internationalization (i18n) Module for GAN Cyber Range.

Provides comprehensive multi-language support for global deployments.
Supports: English, Spanish, French, German, Japanese, Chinese (Simplified).

Global-First Implementation Features:
- Dynamic language switching
- RTL language support  
- Locale-aware formatting
- Translation validation
- Performance-optimized loading
"""

import json
import logging
from typing import Dict, Optional, Any, List
from pathlib import Path
from dataclasses import dataclass
import threading
import time
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class LocaleInfo:
    """Information about a supported locale."""
    code: str
    name: str
    native_name: str
    rtl: bool = False
    plural_rules: str = "default"


class TranslationManager:
    """
    High-performance translation manager with caching and validation.
    
    Features:
    - Thread-safe translation loading
    - Automatic fallback to English
    - Pluralization support
    - Variable interpolation
    - Translation validation
    """
    
    SUPPORTED_LOCALES = {
        'en': LocaleInfo('en', 'English', 'English', False, 'english'),
        'es': LocaleInfo('es', 'Spanish', 'Español', False, 'spanish'),
        'fr': LocaleInfo('fr', 'French', 'Français', False, 'french'),
        'de': LocaleInfo('de', 'German', 'Deutsch', False, 'german'),
        'ja': LocaleInfo('ja', 'Japanese', '日本語', False, 'japanese'),
        'zh-CN': LocaleInfo('zh-CN', 'Chinese (Simplified)', '简体中文', False, 'chinese')
    }
    
    def __init__(self, translations_dir: Optional[Path] = None):
        self.translations_dir = translations_dir or Path(__file__).parent / "translations"
        self.current_locale = 'en'
        self.translations: Dict[str, Dict[str, Any]] = {}
        self.fallback_locale = 'en'
        self._lock = threading.RLock()
        
        # Initialize with built-in translations
        self._initialize_builtin_translations()
        
        # Load external translations if directory exists
        if self.translations_dir.exists():
            self._load_external_translations()
    
    def _initialize_builtin_translations(self):
        """Initialize built-in translations for core functionality."""
        
        # English (default)
        self.translations['en'] = {
            'app': {
                'name': 'GAN Cyber Range Simulator',
                'description': 'Advanced Adversarial Cybersecurity Training Platform'
            },
            'common': {
                'yes': 'Yes',
                'no': 'No',
                'ok': 'OK',
                'cancel': 'Cancel',
                'error': 'Error',
                'warning': 'Warning',
                'info': 'Information',
                'success': 'Success',
                'loading': 'Loading...',
                'save': 'Save',
                'delete': 'Delete',
                'edit': 'Edit',
                'view': 'View',
                'close': 'Close',
                'help': 'Help'
            },
            'security': {
                'threat_detected': 'Threat Detected',
                'threat_blocked': 'Threat Blocked',
                'attack_in_progress': 'Attack in Progress',
                'defense_activated': 'Defense Activated',
                'vulnerability_found': 'Vulnerability Found',
                'patch_applied': 'Patch Applied',
                'security_scan': 'Security Scan',
                'access_denied': 'Access Denied',
                'authenticated': 'Authenticated',
                'unauthorized': 'Unauthorized'
            },
            'agents': {
                'red_team': 'Red Team',
                'blue_team': 'Blue Team',
                'agent_status': 'Agent Status',
                'attack_successful': 'Attack Successful',
                'attack_failed': 'Attack Failed',
                'defense_successful': 'Defense Successful',
                'defense_failed': 'Defense Failed'
            },
            'errors': {
                'connection_failed': 'Connection failed',
                'timeout': 'Operation timed out',
                'invalid_input': 'Invalid input',
                'permission_denied': 'Permission denied',
                'not_found': 'Resource not found',
                'server_error': 'Server error occurred'
            }
        }
        
        # Spanish
        self.translations['es'] = {
            'app': {
                'name': 'Simulador GAN de Campo Cibernético',
                'description': 'Plataforma Avanzada de Entrenamiento de Ciberseguridad Adversarial'
            },
            'common': {
                'yes': 'Sí',
                'no': 'No',
                'ok': 'OK',
                'cancel': 'Cancelar',
                'error': 'Error',
                'warning': 'Advertencia',
                'info': 'Información',
                'success': 'Éxito',
                'loading': 'Cargando...',
                'save': 'Guardar',
                'delete': 'Eliminar',
                'edit': 'Editar',
                'view': 'Ver',
                'close': 'Cerrar',
                'help': 'Ayuda'
            },
            'security': {
                'threat_detected': 'Amenaza Detectada',
                'threat_blocked': 'Amenaza Bloqueada',
                'attack_in_progress': 'Ataque en Progreso',
                'defense_activated': 'Defensa Activada',
                'vulnerability_found': 'Vulnerabilidad Encontrada',
                'patch_applied': 'Parche Aplicado',
                'security_scan': 'Escaneo de Seguridad',
                'access_denied': 'Acceso Denegado',
                'authenticated': 'Autenticado',
                'unauthorized': 'No Autorizado'
            },
            'agents': {
                'red_team': 'Equipo Rojo',
                'blue_team': 'Equipo Azul',
                'agent_status': 'Estado del Agente',
                'attack_successful': 'Ataque Exitoso',
                'attack_failed': 'Ataque Fallido',
                'defense_successful': 'Defensa Exitosa',
                'defense_failed': 'Defensa Fallida'
            },
            'errors': {
                'connection_failed': 'Falló la conexión',
                'timeout': 'Tiempo de espera agotado',
                'invalid_input': 'Entrada inválida',
                'permission_denied': 'Permiso denegado',
                'not_found': 'Recurso no encontrado',
                'server_error': 'Ocurrió un error del servidor'
            }
        }
        
        # French
        self.translations['fr'] = {
            'app': {
                'name': 'Simulateur GAN de Champ Cyber',
                'description': 'Plateforme Avancée de Formation en Cybersécurité Adversariale'
            },
            'common': {
                'yes': 'Oui',
                'no': 'Non',
                'ok': 'OK',
                'cancel': 'Annuler',
                'error': 'Erreur',
                'warning': 'Avertissement',
                'info': 'Information',
                'success': 'Succès',
                'loading': 'Chargement...',
                'save': 'Enregistrer',
                'delete': 'Supprimer',
                'edit': 'Modifier',
                'view': 'Voir',
                'close': 'Fermer',
                'help': 'Aide'
            },
            'security': {
                'threat_detected': 'Menace Détectée',
                'threat_blocked': 'Menace Bloquée',
                'attack_in_progress': 'Attaque en Cours',
                'defense_activated': 'Défense Activée',
                'vulnerability_found': 'Vulnérabilité Trouvée',
                'patch_applied': 'Correctif Appliqué',
                'security_scan': 'Analyse de Sécurité',
                'access_denied': 'Accès Refusé',
                'authenticated': 'Authentifié',
                'unauthorized': 'Non Autorisé'
            },
            'agents': {
                'red_team': 'Équipe Rouge',
                'blue_team': 'Équipe Bleue',
                'agent_status': 'Statut de l\'Agent',
                'attack_successful': 'Attaque Réussie',
                'attack_failed': 'Attaque Échouée',
                'defense_successful': 'Défense Réussie',
                'defense_failed': 'Défense Échouée'
            },
            'errors': {
                'connection_failed': 'Échec de la connexion',
                'timeout': 'Délai d\'attente dépassé',
                'invalid_input': 'Entrée invalide',
                'permission_denied': 'Permission refusée',
                'not_found': 'Ressource introuvable',
                'server_error': 'Une erreur serveur s\'est produite'
            }
        }
        
        # German
        self.translations['de'] = {
            'app': {
                'name': 'GAN Cyber Range Simulator',
                'description': 'Fortgeschrittene Adversariale Cybersicherheits-Trainingsplattform'
            },
            'common': {
                'yes': 'Ja',
                'no': 'Nein',
                'ok': 'OK',
                'cancel': 'Abbrechen',
                'error': 'Fehler',
                'warning': 'Warnung',
                'info': 'Information',
                'success': 'Erfolg',
                'loading': 'Laden...',
                'save': 'Speichern',
                'delete': 'Löschen',
                'edit': 'Bearbeiten',
                'view': 'Anzeigen',
                'close': 'Schließen',
                'help': 'Hilfe'
            },
            'security': {
                'threat_detected': 'Bedrohung Erkannt',
                'threat_blocked': 'Bedrohung Blockiert',
                'attack_in_progress': 'Angriff im Gange',
                'defense_activated': 'Verteidigung Aktiviert',
                'vulnerability_found': 'Schwachstelle Gefunden',
                'patch_applied': 'Patch Angewendet',
                'security_scan': 'Sicherheitsscan',
                'access_denied': 'Zugriff Verweigert',
                'authenticated': 'Authentifiziert',
                'unauthorized': 'Nicht Autorisiert'
            },
            'agents': {
                'red_team': 'Rotes Team',
                'blue_team': 'Blaues Team',
                'agent_status': 'Agent-Status',
                'attack_successful': 'Angriff Erfolgreich',
                'attack_failed': 'Angriff Fehlgeschlagen',
                'defense_successful': 'Verteidigung Erfolgreich',
                'defense_failed': 'Verteidigung Fehlgeschlagen'
            },
            'errors': {
                'connection_failed': 'Verbindung fehlgeschlagen',
                'timeout': 'Zeitüberschreitung',
                'invalid_input': 'Ungültige Eingabe',
                'permission_denied': 'Berechtigung verweigert',
                'not_found': 'Ressource nicht gefunden',
                'server_error': 'Serverfehler aufgetreten'
            }
        }
        
        # Japanese
        self.translations['ja'] = {
            'app': {
                'name': 'GANサイバーレンジシミュレーター',
                'description': '高度な敵対的サイバーセキュリティトレーニングプラットフォーム'
            },
            'common': {
                'yes': 'はい',
                'no': 'いいえ',
                'ok': 'OK',
                'cancel': 'キャンセル',
                'error': 'エラー',
                'warning': '警告',
                'info': '情報',
                'success': '成功',
                'loading': '読み込み中...',
                'save': '保存',
                'delete': '削除',
                'edit': '編集',
                'view': '表示',
                'close': '閉じる',
                'help': 'ヘルプ'
            },
            'security': {
                'threat_detected': '脅威を検出',
                'threat_blocked': '脅威をブロック',
                'attack_in_progress': '攻撃進行中',
                'defense_activated': '防御を有効化',
                'vulnerability_found': '脆弱性を発見',
                'patch_applied': 'パッチを適用',
                'security_scan': 'セキュリティスキャン',
                'access_denied': 'アクセス拒否',
                'authenticated': '認証済み',
                'unauthorized': '権限なし'
            },
            'agents': {
                'red_team': 'レッドチーム',
                'blue_team': 'ブルーチーム',
                'agent_status': 'エージェント状態',
                'attack_successful': '攻撃成功',
                'attack_failed': '攻撃失敗',
                'defense_successful': '防御成功',
                'defense_failed': '防御失敗'
            },
            'errors': {
                'connection_failed': '接続に失敗しました',
                'timeout': 'タイムアウトしました',
                'invalid_input': '無効な入力です',
                'permission_denied': '権限が拒否されました',
                'not_found': 'リソースが見つかりません',
                'server_error': 'サーバーエラーが発生しました'
            }
        }
        
        # Chinese (Simplified)
        self.translations['zh-CN'] = {
            'app': {
                'name': 'GAN网络靶场模拟器',
                'description': '高级对抗式网络安全训练平台'
            },
            'common': {
                'yes': '是',
                'no': '否',
                'ok': '确定',
                'cancel': '取消',
                'error': '错误',
                'warning': '警告',
                'info': '信息',
                'success': '成功',
                'loading': '加载中...',
                'save': '保存',
                'delete': '删除',
                'edit': '编辑',
                'view': '查看',
                'close': '关闭',
                'help': '帮助'
            },
            'security': {
                'threat_detected': '检测到威胁',
                'threat_blocked': '威胁已阻止',
                'attack_in_progress': '攻击进行中',
                'defense_activated': '防御已激活',
                'vulnerability_found': '发现漏洞',
                'patch_applied': '补丁已应用',
                'security_scan': '安全扫描',
                'access_denied': '访问被拒绝',
                'authenticated': '已认证',
                'unauthorized': '未授权'
            },
            'agents': {
                'red_team': '红队',
                'blue_team': '蓝队',
                'agent_status': '代理状态',
                'attack_successful': '攻击成功',
                'attack_failed': '攻击失败',
                'defense_successful': '防御成功',
                'defense_failed': '防御失败'
            },
            'errors': {
                'connection_failed': '连接失败',
                'timeout': '操作超时',
                'invalid_input': '输入无效',
                'permission_denied': '权限被拒绝',
                'not_found': '资源未找到',
                'server_error': '服务器错误'
            }
        }
        
        logger.info(f"Initialized built-in translations for {len(self.translations)} locales")
    
    def _load_external_translations(self):
        """Load external translation files from filesystem."""
        try:
            for locale_code in self.SUPPORTED_LOCALES.keys():
                translation_file = self.translations_dir / f"{locale_code}.json"
                
                if translation_file.exists():
                    with open(translation_file, 'r', encoding='utf-8') as f:
                        external_translations = json.load(f)
                    
                    # Merge with built-in translations
                    if locale_code in self.translations:
                        self._deep_merge_dict(self.translations[locale_code], external_translations)
                    else:
                        self.translations[locale_code] = external_translations
                    
                    logger.info(f"Loaded external translations for {locale_code}")
        
        except Exception as e:
            logger.warning(f"Failed to load external translations: {e}")
    
    def _deep_merge_dict(self, base: dict, update: dict):
        """Deep merge two dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge_dict(base[key], value)
            else:
                base[key] = value
    
    def set_locale(self, locale_code: str) -> bool:
        """
        Set the current locale.
        
        Args:
            locale_code: Language code (e.g., 'en', 'es', 'fr')
            
        Returns:
            True if locale was set successfully
        """
        with self._lock:
            if locale_code in self.SUPPORTED_LOCALES:
                old_locale = self.current_locale
                self.current_locale = locale_code
                logger.info(f"Locale changed from {old_locale} to {locale_code}")
                return True
            else:
                logger.warning(f"Unsupported locale: {locale_code}")
                return False
    
    def get_locale(self) -> str:
        """Get the current locale code."""
        return self.current_locale
    
    def get_locale_info(self, locale_code: Optional[str] = None) -> LocaleInfo:
        """Get information about a locale."""
        code = locale_code or self.current_locale
        return self.SUPPORTED_LOCALES.get(code, self.SUPPORTED_LOCALES['en'])
    
    def get_supported_locales(self) -> List[LocaleInfo]:
        """Get list of all supported locales."""
        return list(self.SUPPORTED_LOCALES.values())
    
    @lru_cache(maxsize=1000)
    def translate(self, key: str, locale: Optional[str] = None, **kwargs) -> str:
        """
        Translate a key to the specified locale.
        
        Args:
            key: Translation key (e.g., 'common.yes', 'security.threat_detected')
            locale: Target locale (uses current locale if None)
            **kwargs: Variables for string interpolation
            
        Returns:
            Translated string
        """
        target_locale = locale or self.current_locale
        
        # Get translation from target locale or fallback
        translation = self._get_translation(key, target_locale)
        
        # Perform variable interpolation if needed
        if kwargs and isinstance(translation, str):
            try:
                translation = translation.format(**kwargs)
            except (KeyError, ValueError) as e:
                logger.warning(f"Translation interpolation failed for key '{key}': {e}")
        
        return translation
    
    def _get_translation(self, key: str, locale: str) -> str:
        """Get translation for a key, with fallback logic."""
        # Try target locale first
        translation = self._lookup_translation(key, locale)
        if translation is not None:
            return translation
        
        # Fallback to English if different
        if locale != self.fallback_locale:
            translation = self._lookup_translation(key, self.fallback_locale)
            if translation is not None:
                logger.debug(f"Using fallback translation for key '{key}' (locale: {locale})")
                return translation
        
        # Return key if no translation found
        logger.warning(f"No translation found for key '{key}' (locale: {locale})")
        return key
    
    def _lookup_translation(self, key: str, locale: str) -> Optional[str]:
        """Look up a translation in the specified locale."""
        if locale not in self.translations:
            return None
        
        # Navigate nested dictionary using key path
        current = self.translations[locale]
        for part in key.split('.'):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current if isinstance(current, str) else None
    
    def format_number(self, number: float, locale: Optional[str] = None) -> str:
        """Format number according to locale conventions."""
        target_locale = locale or self.current_locale
        locale_info = self.get_locale_info(target_locale)
        
        # Basic number formatting (can be enhanced with locale-specific rules)
        if target_locale in ['en']:
            return f"{number:,.2f}"
        elif target_locale in ['de', 'fr']:
            return f"{number:,.2f}".replace(',', ' ').replace('.', ',')
        elif target_locale in ['es']:
            return f"{number:,.2f}".replace(',', '.')
        else:
            return f"{number:.2f}"
    
    def format_datetime(self, dt, format_type: str = 'short', locale: Optional[str] = None) -> str:
        """Format datetime according to locale conventions."""
        target_locale = locale or self.current_locale
        
        # Basic datetime formatting (can be enhanced with proper locale support)
        if format_type == 'short':
            if target_locale in ['en']:
                return dt.strftime("%m/%d/%Y %I:%M %p")
            elif target_locale in ['de']:
                return dt.strftime("%d.%m.%Y %H:%M")
            elif target_locale in ['fr']:
                return dt.strftime("%d/%m/%Y %H:%M")
            elif target_locale in ['es']:
                return dt.strftime("%d/%m/%Y %H:%M")
            elif target_locale in ['ja']:
                return dt.strftime("%Y/%m/%d %H:%M")
            elif target_locale in ['zh-CN']:
                return dt.strftime("%Y-%m-%d %H:%M")
        
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    def validate_translations(self) -> Dict[str, List[str]]:
        """Validate translations and return missing keys."""
        validation_results = {}
        
        # Use English as reference
        english_keys = self._get_all_keys(self.translations.get('en', {}))
        
        for locale_code in self.SUPPORTED_LOCALES.keys():
            if locale_code == 'en':
                continue
                
            locale_keys = self._get_all_keys(self.translations.get(locale_code, {}))
            missing_keys = english_keys - locale_keys
            
            if missing_keys:
                validation_results[locale_code] = sorted(missing_keys)
        
        return validation_results
    
    def _get_all_keys(self, translations: dict, prefix: str = '') -> set:
        """Get all translation keys from nested dictionary."""
        keys = set()
        
        for key, value in translations.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                keys.update(self._get_all_keys(value, full_key))
            else:
                keys.add(full_key)
        
        return keys


# Global translation manager instance
_translation_manager = None
_translation_manager_lock = threading.Lock()


def get_translation_manager() -> TranslationManager:
    """Get the global translation manager instance (thread-safe singleton)."""
    global _translation_manager
    
    if _translation_manager is None:
        with _translation_manager_lock:
            if _translation_manager is None:
                _translation_manager = TranslationManager()
    
    return _translation_manager


def t(key: str, locale: Optional[str] = None, **kwargs) -> str:
    """
    Convenient translation function.
    
    Args:
        key: Translation key
        locale: Target locale (optional)
        **kwargs: Variables for string interpolation
        
    Returns:
        Translated string
    """
    return get_translation_manager().translate(key, locale, **kwargs)


def set_locale(locale_code: str) -> bool:
    """Set the global locale."""
    return get_translation_manager().set_locale(locale_code)


def get_locale() -> str:
    """Get the current global locale."""
    return get_translation_manager().get_locale()


def get_supported_locales() -> List[LocaleInfo]:
    """Get list of supported locales."""
    return get_translation_manager().get_supported_locales()


# Compliance helpers for global deployment
class ComplianceHelper:
    """Helper for GDPR, CCPA, PDPA compliance in different regions."""
    
    REGION_REGULATIONS = {
        'EU': ['GDPR'],
        'US': ['CCPA', 'SOX'],
        'CA': ['PIPEDA'],
        'SG': ['PDPA'],
        'JP': ['APPI'],
        'CN': ['PIPL', 'CSL']
    }
    
    @staticmethod
    def get_privacy_notice(locale: str) -> str:
        """Get localized privacy notice."""
        return t('compliance.privacy_notice', locale)
    
    @staticmethod
    def get_data_retention_policy(region: str) -> Dict[str, int]:
        """Get data retention policy for region."""
        policies = {
            'EU': {'logs': 90, 'metrics': 365, 'personal_data': 1095},
            'US': {'logs': 180, 'metrics': 1095, 'personal_data': 2555},
            'default': {'logs': 90, 'metrics': 365, 'personal_data': 1095}
        }
        return policies.get(region, policies['default'])


if __name__ == "__main__":
    # Example usage and testing
    tm = TranslationManager()
    
    # Test basic translation
    print("English:", tm.translate('common.yes'))
    
    # Test locale switching
    tm.set_locale('es')
    print("Spanish:", tm.translate('common.yes'))
    
    tm.set_locale('ja')
    print("Japanese:", tm.translate('common.yes'))
    
    # Test variable interpolation
    tm.set_locale('en')
    message = tm.translate('security.threat_detected')
    print(f"Security message: {message}")
    
    # Validate translations
    validation = tm.validate_translations()
    print(f"Translation validation: {validation}")