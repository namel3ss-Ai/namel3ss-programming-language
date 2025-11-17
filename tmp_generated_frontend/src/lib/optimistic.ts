export interface OptimisticMutation<T extends Record<string, unknown>> {
  id?: string;
  apply: (state: T) => T;
  rollback?: (state: T) => T;
}

export interface PreparedOptimisticMutation<T extends Record<string, unknown>> extends OptimisticMutation<T> {
  id: string;
}

const randomId = (): string => {
  const globalCrypto = typeof globalThis !== "undefined" ? (globalThis as { crypto?: Crypto }).crypto : undefined;
  if (globalCrypto && typeof globalCrypto.randomUUID === "function") {
    return globalCrypto.randomUUID();
  }
  return Math.random().toString(36).slice(2);
};

export function prepareOptimisticMutation<T extends Record<string, unknown>>(
  mutation: OptimisticMutation<T>,
): PreparedOptimisticMutation<T> {
  if (typeof mutation.apply !== "function") {
    throw new Error("Optimistic mutation requires an apply function.");
  }
  const id = mutation.id ?? randomId();
  return { ...mutation, id };
}

export function applyPreparedOptimistic<T extends Record<string, unknown>>(
  state: Record<string, unknown>,
  mutation: PreparedOptimisticMutation<T>,
): Record<string, unknown> {
  const draft = { ...state } as T;
  return mutation.apply(draft) ?? draft;
}

export function rollbackPreparedOptimistic<T extends Record<string, unknown>>(
  state: Record<string, unknown>,
  mutation: PreparedOptimisticMutation<T>,
): Record<string, unknown> {
  if (typeof mutation.rollback === "function") {
    const draft = { ...state } as T;
    return mutation.rollback(draft) ?? draft;
  }
  return state;
}
