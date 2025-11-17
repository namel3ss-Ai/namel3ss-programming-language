import { FormEvent, useMemo, useRef, useState } from "react";
import type { FormWidgetConfig, PageDataReloadOptions, RequestError } from "../lib/n3Client";
import { submitJson } from "../lib/n3Client";
import ErrorBanner from "./ErrorBanner";
import { InlineSpinner } from "./LoadingState";
import { useToast } from "./Toast";

interface FormWidgetProps {
  widget: FormWidgetConfig;
  pageSlug: string;
  fieldErrors?: Record<string, string[]>;
  optimisticTargets?: readonly string[];
  onReload: (options?: PageDataReloadOptions) => Promise<void>;
  applyOptimistic: (mutation: OptimisticMutation<Record<string, unknown>>) => string;
  rollbackOptimistic: (id?: string) => void;
  clearOptimistic: () => void;
}

export default function FormWidget({
  widget,
  pageSlug,
  fieldErrors,
  optimisticTargets,
  onReload,
  applyOptimistic,
  rollbackOptimistic,
  clearOptimistic,
}: FormWidgetProps) {
  const toast = useToast();
  const [submitting, setSubmitting] = useState(false);
  const [submissionError, setSubmissionError] = useState<string | null>(null);
  const [localFieldErrors, setLocalFieldErrors] = useState<Record<string, string[]>>({});
  const optimisticIdRef = useRef<string | null>(null);

  const combinedFieldErrors = useMemo(() => {
    const result: Record<string, string[]> = {};
    const sources = [fieldErrors ?? {}, localFieldErrors];
    sources.forEach((source) => {
      Object.entries(source).forEach(([name, messages]) => {
        if (!Array.isArray(messages)) {
          return;
        }
        if (!result[name]) {
          result[name] = [];
        }
        messages.forEach((entry) => {
          if (entry && !result[name].includes(entry)) {
            result[name].push(entry);
          }
        });
      });
    });
    return result;
  }, [fieldErrors, localFieldErrors]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const payload: Record<string, unknown> = {};
    form.forEach((value, key) => {
      payload[key] = value;
    });

    setSubmitting(true);
    setSubmissionError(null);
    setLocalFieldErrors({});

    const targetIds = Array.isArray(optimisticTargets) ? optimisticTargets : [];

    const optimisticId = applyOptimistic({
      apply: (state) => {
        const draft = { ...state } as Record<string, unknown>;
        const baseOptimistic = (draft.optimistic as Record<string, unknown> | undefined) ?? {};
        const bucket: Record<string, unknown> = { ...baseOptimistic };
        bucket[widget.id] = { pending: true };
        targetIds.forEach((targetId) => {
          const existing = bucket[targetId];
          if (existing && typeof existing === "object" && !Array.isArray(existing)) {
            bucket[targetId] = { ...(existing as Record<string, unknown>), pending: true };
          } else {
            bucket[targetId] = { pending: true };
          }
        });
        draft.optimistic = bucket;
        return draft;
      },
      rollback: (state) => {
        const draft = { ...state } as Record<string, unknown>;
        if (draft.optimistic && typeof draft.optimistic === "object") {
          const bucket = { ...(draft.optimistic as Record<string, unknown>) };
          delete bucket[widget.id];
          targetIds.forEach((targetId) => {
            if (!(targetId in bucket)) {
              return;
            }
            const entry = bucket[targetId];
            if (entry && typeof entry === "object" && !Array.isArray(entry)) {
              const { pending: _pending, ...rest } = entry as Record<string, unknown>;
              if (Object.keys(rest).length) {
                bucket[targetId] = rest;
              } else {
                delete bucket[targetId];
              }
            } else {
              delete bucket[targetId];
            }
          });
          draft.optimistic = bucket;
        }
        return draft;
      },
    });
    optimisticIdRef.current = optimisticId;

    try {
      await submitJson(`/api/pages/${pageSlug}/forms/${widget.id}`, payload, {
        credentials: "include",
      });
      clearOptimistic();
      toast.show(widget.successMessage ?? "Form submitted");
      await onReload({ silent: true });
    } catch (error) {
      rollbackOptimistic(optimisticIdRef.current ?? undefined);
      const requestError = error as RequestError;
      if (requestError.fieldErrors) {
        setLocalFieldErrors(requestError.fieldErrors);
      }
      setSubmissionError(requestError.message ?? "Unable to submit form right now.");
    } finally {
      setSubmitting(false);
      optimisticIdRef.current = null;
    }
  }

  return (
    <section className="n3-widget" aria-busy={submitting} aria-live="polite">
      <h3>{widget.title}</h3>
      {submissionError ? <ErrorBanner errors={[submissionError]} tone="warning" /> : null}
      <form onSubmit={handleSubmit} style={{ display: "grid", gap: "0.75rem", maxWidth: "420px" }} noValidate>
        {widget.fields.map((field) => {
          const errors = combinedFieldErrors[field.name] ?? [];
          return (
            <label key={field.name} style={{ display: "grid", gap: "0.35rem" }}>
              <span style={{ fontWeight: 600 }}>{field.name}</span>
              <input
                name={field.name}
                type={field.type ?? "text"}
                required
                aria-invalid={errors.length > 0}
                aria-describedby={errors.length ? `${widget.id}-${field.name}-errors` : undefined}
                style={{
                  padding: "0.55rem 0.75rem",
                  borderRadius: "0.5rem",
                  border: errors.length ? "1px solid var(--warning, #dc2626)" : "1px solid rgba(15,23,42,0.18)",
                  outline: errors.length ? "2px solid rgba(220,38,38,0.18)" : undefined,
                }}
              />
              {errors.length ? (
                <ul id={`${widget.id}-${field.name}-errors`} style={{ margin: 0, paddingInlineStart: "1rem", color: "var(--warning, #dc2626)", fontSize: "0.85rem" }}>
                  {errors.map((message, index) => (
                    <li key={index}>{message}</li>
                  ))}
                </ul>
              ) : null}
            </label>
          );
        })}
        <button
          type="submit"
          disabled={submitting}
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "0.5rem",
            padding: "0.65rem 1.25rem",
            borderRadius: "0.65rem",
            border: "none",
            background: "var(--primary, #2563eb)",
            color: "#fff",
            fontWeight: 600,
          }}
        >
          {submitting ? (
            <>
              <InlineSpinner />
              <span>Submitting...</span>
            </>
          ) : (
            <span>Submit</span>
          )}
        </button>
      </form>
    </section>
  );
}
