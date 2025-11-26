import { useEffect, useState } from "react";

export interface DataSourceRef {
  kind: string;
  name: string;
}

export interface TextWidgetConfig {
  id: string;
  type: "text";
  text: string;
  styles?: Record<string, string>;
}

export interface TableWidgetConfig {
  id: string;
  type: "table";
  title: string;
  columns: string[];
  source: DataSourceRef;
}

export interface ChartWidgetConfig {
  id: string;
  type: "chart";
  title: string;
  chartType?: string;
  source: DataSourceRef;
  x?: string | null;
  y?: string | null;
}

export interface FormWidgetField {
  name: string;
  type?: string;
}

export interface FormWidgetConfig {
  id: string;
  type: "form";
  title: string;
  fields: FormWidgetField[];
  successMessage?: string | null;
}

export interface CardWidgetConfig {
  id: string;
  type: "card";
  title: string;
  source: DataSourceRef;
  emptyState?: {
    icon?: string;
    iconSize?: "small" | "medium" | "large";
    title: string;
    message?: string;
    actionLabel?: string;
    actionLink?: string;
  };
  itemConfig?: Record<string, any>;
  groupBy?: string;
}

export interface ListWidgetConfig {
  id: string;
  type: "list";
  title: string;
  source: DataSourceRef;
  listType?: string;
  emptyState?: Record<string, any>;
  itemConfig?: Record<string, any>;
  enableSearch?: boolean;
  columns?: number;
}

export type WidgetConfig =
  | TextWidgetConfig
  | TableWidgetConfig
  | ChartWidgetConfig
  | FormWidgetConfig
  | CardWidgetConfig
  | ListWidgetConfig;

export interface PageDefinition {
  slug: string;
  route: string;
  title: string;
  description?: string | null;
  reactive: boolean;
  realtime: boolean;
  widgets: WidgetConfig[];
  preview: Record<string, unknown>;
}

export interface PageDataState {
  data: Record<string, unknown> | null;
  loading: boolean;
  error: string | null;
}

type UnknownRecord = Record<string, any>;

function isRecord(value: unknown): value is UnknownRecord {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function cloneObjectArray(value: unknown, deep?: boolean): UnknownRecord[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return (value as UnknownRecord[]).map((item) => {
    const source = isRecord(item) ? item : {};
    const clone: UnknownRecord = { ...source };
    if (deep) {
      if (Array.isArray(source.rows)) {
        clone.rows = cloneObjectArray(source.rows, true);
      }
      if (isRecord(source.summary)) {
        clone.summary = { ...source.summary };
      }
      if (isRecord(source.metadata)) {
        clone.metadata = { ...source.metadata };
      }
    }
    return clone;
  });
}

function cloneDataset(dataset: UnknownRecord): UnknownRecord {
  const clone: UnknownRecord = { ...dataset };
  if (Array.isArray(dataset.rows)) {
    clone.rows = cloneObjectArray(dataset.rows, true);
  }
  if (isRecord(dataset.summary)) {
    clone.summary = { ...dataset.summary };
  }
  if (isRecord(dataset.metadata)) {
    clone.metadata = { ...dataset.metadata };
  }
  return clone;
}

function normaliseRowId(row: UnknownRecord): string | null {
  const candidate = row.id ?? row.rowId ?? row.__id;
  if (candidate === undefined || candidate === null) {
    return null;
  }
  return String(candidate);
}

function upsertRows(base: UnknownRecord[], patches: UnknownRecord[]): UnknownRecord[] {
  if (!patches.length) {
    return base;
  }
  const result = base.map((row) => ({ ...row }));
  const byId = new Map<string, UnknownRecord>();
  result.forEach((row) => {
    const id = normaliseRowId(row);
    if (id !== null) {
      byId.set(id, row);
    }
  });
  patches.forEach((patch) => {
    const id = normaliseRowId(patch);
    if (id === null) {
      result.push({ ...patch });
      return;
    }
    const target = byId.get(id);
    if (target) {
      Object.assign(target, patch);
    } else {
      const clone = { ...patch };
      byId.set(id, clone);
      result.push(clone);
    }
  });
  return result;
}

function applyRowReplacements(base: UnknownRecord[], replacements: UnknownRecord): UnknownRecord[] {
  const keys = Object.keys(replacements ?? {});
  if (!keys.length) {
    return base;
  }
  const result = base.map((row) => ({ ...row }));
  const byId = new Map<string, UnknownRecord>();
  result.forEach((row) => {
    const id = normaliseRowId(row);
    if (id !== null) {
      byId.set(id, row);
    }
  });
  keys.forEach((key) => {
    const patch = replacements[key];
    if (!isRecord(patch)) {
      return;
    }
    const target = byId.get(key);
    if (target) {
      Object.assign(target, patch);
    } else {
      const clone: UnknownRecord = { id: key, ...patch };
      byId.set(key, clone);
      result.push(clone);
    }
  });
  return result;
}

function appendRowCollection(base: UnknownRecord[], additions: UnknownRecord[]): UnknownRecord[] {
  if (!additions.length) {
    return base;
  }
  return [...base, ...additions.map((row) => ({ ...row }))];
}

function mergeRowOverlays(baseRows: unknown, overlay: UnknownRecord): UnknownRecord[] | undefined {
  const hasDirectRows = Array.isArray(overlay.rows);
  let rows = hasDirectRows ? cloneObjectArray(overlay.rows) : Array.isArray(baseRows) ? cloneObjectArray(baseRows) : [];
  let mutated = hasDirectRows;

  const optimisticRows = cloneObjectArray(overlay.optimisticRows);
  if (optimisticRows.length) {
    rows = upsertRows(rows, optimisticRows);
    mutated = true;
  }

  const replaceRows = isRecord(overlay.replaceRows) ? overlay.replaceRows : {};
  if (Object.keys(replaceRows).length) {
    rows = applyRowReplacements(rows, replaceRows);
    mutated = true;
  }

  const appendRows = cloneObjectArray(overlay.appendRows);
  if (appendRows.length) {
    rows = appendRowCollection(rows, appendRows);
    mutated = true;
  }

  return mutated ? rows : undefined;
}

function mergeSummary(baseSummary: unknown, overlay: unknown): UnknownRecord | undefined {
  const direct = isRecord(overlay) ? overlay : undefined;
  if (!direct) {
    return isRecord(baseSummary) ? { ...baseSummary } : undefined;
  }
  const base = isRecord(baseSummary) ? { ...baseSummary } : {};
  Object.assign(base, direct);
  return base;
}

function mergeMetadata(baseMetadata: unknown, overlay: unknown): UnknownRecord | undefined {
  const direct = isRecord(overlay) ? overlay : undefined;
  if (!direct) {
    return isRecord(baseMetadata) ? { ...baseMetadata } : undefined;
  }
  const base = isRecord(baseMetadata) ? { ...baseMetadata } : {};
  Object.assign(base, direct);
  return base;
}

function mergeDatasetOverlays(baseDatasets: unknown, overlays: unknown): UnknownRecord[] | undefined {
  if (!Array.isArray(overlays) || overlays.length === 0) {
    return undefined;
  }
  const result = Array.isArray(baseDatasets)
    ? (baseDatasets as UnknownRecord[]).map((dataset) => cloneDataset(dataset))
    : [];
  const index = new Map<string, UnknownRecord>();
  result.forEach((dataset) => {
    const key = dataset.id ?? dataset.name;
    if (key !== undefined && key !== null) {
      index.set(String(key), dataset);
    }
  });
  (overlays as UnknownRecord[]).forEach((candidate) => {
    if (!isRecord(candidate)) {
      return;
    }
    const datasetPatch = candidate as UnknownRecord;
    const keyValue = datasetPatch.id ?? datasetPatch.name;
    if (keyValue === undefined || keyValue === null) {
      result.push(cloneDataset(datasetPatch));
      return;
    }
    const key = String(keyValue);
    const target = index.get(key);
    if (!target) {
      const clone = cloneDataset(datasetPatch);
      index.set(key, clone);
      result.push(clone);
      return;
    }
    const rowOverlay = mergeRowOverlays(target.rows, datasetPatch);
    if (rowOverlay) {
      target.rows = rowOverlay;
    }
    const summaryOverlay = mergeSummary(target.summary, datasetPatch.optimisticSummary ?? datasetPatch.summary);
    if (summaryOverlay) {
      target.summary = summaryOverlay;
    }
    const metadataOverlay = mergeMetadata(target.metadata, datasetPatch.metadata);
    if (metadataOverlay) {
      target.metadata = metadataOverlay;
    }
    Object.entries(datasetPatch).forEach(([field, value]) => {
      if (
        field === "id" ||
        field === "name" ||
        field === "optimisticRows" ||
        field === "appendRows" ||
        field === "replaceRows" ||
        field === "optimisticSummary" ||
        field === "metadata"
      ) {
        return;
      }
      if (field === "rows" && Array.isArray(value)) {
        target.rows = cloneObjectArray(value);
        return;
      }
      if (field === "summary" && isRecord(value)) {
        target.summary = { ...value };
        return;
      }
      if (field === "metadata" && isRecord(value)) {
        target.metadata = { ...value };
        return;
      }
      target[field] = value;
    });
  });
  return result;
}

function mergeOptimisticData(base: unknown, optimistic: unknown): unknown {
  if (optimistic === undefined || optimistic === null) {
    return base;
  }
  if (!isRecord(optimistic)) {
    return optimistic;
  }
  const patch = optimistic as UnknownRecord;
  const baseRecord: UnknownRecord = isRecord(base) ? { ...base } : {};
  let mutated = false;
  if (Object.prototype.hasOwnProperty.call(patch, "pending")) {
    baseRecord.pending = patch.pending;
    mutated = true;
  }
  if (Object.prototype.hasOwnProperty.call(patch, "error")) {
    baseRecord.error = patch.error;
    mutated = true;
  }
  const dataOverlay = isRecord(patch.data) ? (patch.data as UnknownRecord) : undefined;
  if (dataOverlay) {
    const rowOverlay = mergeRowOverlays(baseRecord.rows, dataOverlay);
    if (rowOverlay) {
      baseRecord.rows = rowOverlay;
      mutated = true;
    }
    const datasetsOverlay = mergeDatasetOverlays(baseRecord.datasets, dataOverlay.optimisticDatasets);
    if (datasetsOverlay) {
      baseRecord.datasets = datasetsOverlay;
      mutated = true;
    }
    const summaryOverlay = mergeSummary(baseRecord.summary, dataOverlay.optimisticSummary ?? dataOverlay.summary);
    if (summaryOverlay) {
      baseRecord.summary = summaryOverlay;
      mutated = true;
    }
    const metadataOverlay = mergeMetadata(baseRecord.metadata, dataOverlay.metadata);
    if (metadataOverlay) {
      baseRecord.metadata = metadataOverlay;
      mutated = true;
    }
    Object.entries(dataOverlay).forEach(([field, value]) => {
      if (
        field === "optimisticRows" ||
        field === "appendRows" ||
        field === "replaceRows" ||
        field === "optimisticSummary" ||
        field === "optimisticDatasets" ||
        field === "metadata" ||
        field === "rows" ||
        field === "summary"
      ) {
        return;
      }
      baseRecord[field] = value;
      mutated = true;
    });
  }
  return mutated ? baseRecord : base;
}

export function usePageData(definition: PageDefinition): PageDataState {
  const [state, setState] = useState<PageDataState>({ data: null, loading: true, error: null });

  useEffect(() => {
    let cancelled = false;
    setState({ data: null, loading: true, error: null });
    fetch(`/api/pages/${definition.slug}`, {
      headers: { Accept: "application/json" },
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Request failed: ${response.status}`);
        }
        return response.json();
      })
      .then((payload) => {
        if (!cancelled) {
          setState({ data: payload ?? {}, loading: false, error: null });
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setState({ data: null, loading: false, error: error instanceof Error ? error.message : String(error) });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [definition.slug]);

  return state;
}

export function resolveWidgetData(widgetId: string, pageData: Record<string, unknown> | null | undefined): unknown {
  if (!pageData || typeof pageData !== "object") {
    return undefined;
  }
  const buckets = [
    (pageData as any).widgets,
    (pageData as any).components,
    (pageData as any).data,
  ];
  let resolved: unknown = undefined;
  for (const bucket of buckets) {
    if (bucket && typeof bucket === "object" && widgetId in bucket) {
      resolved = (bucket as Record<string, unknown>)[widgetId];
      break;
    }
  }
  if (resolved === undefined && widgetId in (pageData as Record<string, unknown>)) {
    resolved = (pageData as Record<string, unknown>)[widgetId];
  }
  const optimisticRoot = (pageData as any).optimistic;
  if (!optimisticRoot || typeof optimisticRoot !== "object") {
    return resolved;
  }
  const optimisticEntry = (optimisticRoot as Record<string, unknown>)[widgetId];
  if (optimisticEntry === undefined) {
    return resolved;
  }
  return mergeOptimisticData(resolved, optimisticEntry);
}

export function ensureArray<T>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]) : [];
}
