/*
 * Hyperparameter config types [Wizard Configure step].
 *
 * HTML inputs always hand back strings. The step used to decide the input type
 * from the *current* value, so once a value was a string (or the field had been
 * cleared) every later keystroke stayed a string and run.json got
 * `batch_size: "4"`. The schema decides the type here, and the same coercion
 * runs before the config is posted.
 */

export interface HparamProperty {
  type?: string | string[];
  default?: unknown;
  minimum?: number;
  maximum?: number;
}

export type HparamSchema = { properties?: Record<string, HparamProperty> } | undefined | null;

export function typeList(prop: HparamProperty | undefined): string[] {
  const value = prop?.type;
  if (Array.isArray(value)) return value;
  return value ? [value] : [];
}

export function isNumericProperty(prop: HparamProperty | undefined): boolean {
  const types = typeList(prop);
  return types.includes("integer") || types.includes("number");
}

export function isIntegerProperty(prop: HparamProperty | undefined): boolean {
  return typeList(prop).includes("integer");
}

export function propertiesOf(schema: HparamSchema): Record<string, HparamProperty> {
  const properties = schema?.properties;
  return properties && typeof properties === "object" ? properties : {};
}

/** One field value from the raw input string, using the schema's type. */
export function parseFieldValue(prop: HparamProperty | undefined, raw: string): unknown {
  if (typeList(prop).includes("boolean")) return raw === "true";
  if (isNumericProperty(prop)) {
    if (raw.trim() === "") return "";
    const parsed = Number(raw);
    return Number.isNaN(parsed) ? raw : parsed;
  }
  return raw;
}

/**
 * Final config for the API: numeric fields become numbers, booleans become
 * booleans, and empty numeric fields are dropped so the backend default wins.
 */
export function coerceConfig(
  config: Record<string, unknown>,
  schema: HparamSchema,
): Record<string, unknown> {
  const properties = propertiesOf(schema);
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(config)) {
    const prop = properties[key];
    if (prop === undefined) {
      result[key] = value;
      continue;
    }
    if (isNumericProperty(prop)) {
      if (typeof value === "number" && Number.isFinite(value)) {
        result[key] = isIntegerProperty(prop) ? Math.trunc(value) : value;
      } else if (typeof value === "string" && value.trim() !== "") {
        const parsed = Number(value);
        if (!Number.isNaN(parsed)) {
          result[key] = isIntegerProperty(prop) ? Math.trunc(parsed) : parsed;
        }
      }
      // Empty/NaN: leave the key out so the plugin default applies.
      continue;
    }
    if (typeList(prop).includes("boolean")) {
      result[key] = typeof value === "boolean" ? value : value === "true";
      continue;
    }
    result[key] = value;
  }
  return result;
}
