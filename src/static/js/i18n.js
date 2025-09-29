const I18N_DEFAULT_LOCALE = (() => {
  if (typeof document !== "undefined" && document?.documentElement) {
    return document.documentElement.lang || "pt-BR";
  }
  return "pt-BR";
})();

const I18N_STATE = {
  locale: I18N_DEFAULT_LOCALE,
  translations: {},
};

function getTranslationByKey(key) {
  if (!key) {
    return undefined;
  }
  const parts = String(key).split(".");
  let current = I18N_STATE.translations;
  for (const part of parts) {
    if (current && Object.prototype.hasOwnProperty.call(current, part)) {
      current = current[part];
    } else {
      return undefined;
    }
  }
  return typeof current === "string" ? current : undefined;
}

function translate(key, fallback, vars) {
  const raw = getTranslationByKey(key);
  const base = typeof raw === "string" ? raw : (typeof fallback === "string" ? fallback : String(key));
  if (!vars) {
    return base;
  }
  return base.replace(/\{([^}]+)\}/g, (match, token) => {
    if (Object.prototype.hasOwnProperty.call(vars, token)) {
      const value = vars[token];
      return value == null ? "" : String(value);
    }
    return match;
  });
}

export const t = (key, fallback, vars) => translate(key, fallback, vars);

export async function loadTranslations(locale) {
  const target = locale || I18N_DEFAULT_LOCALE;
  const url = `static/i18n/${target}.json`;
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load translations for ${target}`);
    }
    const data = await response.json();
    I18N_STATE.locale = target;
    I18N_STATE.translations = data || {};
  } catch (err) {
    console.warn("Não foi possível carregar traduções:", err);
    I18N_STATE.locale = target;
    I18N_STATE.translations = {};
  }
}

export { I18N_DEFAULT_LOCALE, I18N_STATE };
