import type { PropsWithChildren, ReactNode } from "react";

export interface ErrorBannerProps {
  errors: readonly (string | { message?: string; detail?: string })[];
  tone?: "error" | "warning" | "info";
  onDismiss?: () => void;
  action?: ReactNode;
}

export default function ErrorBanner({ errors, tone = "error", onDismiss, action, children }: PropsWithChildren<ErrorBannerProps>) {
  if ((!errors || !errors.length) && !children) {
    return null;
  }
  const palette = tone === "warning"
    ? { background: "rgba(245, 158, 11, 0.18)", border: "rgba(245, 158, 11, 0.45)", color: "#92400e" }
    : tone === "info"
      ? { background: "rgba(59, 130, 246, 0.18)", border: "rgba(59, 130, 246, 0.45)", color: "#1e3a8a" }
      : { background: "rgba(220, 38, 38, 0.16)", border: "rgba(220, 38, 38, 0.4)", color: "#991b1b" };

  return (
    <section
      role="alert"
      aria-live="assertive"
      className="n3-error-banner"
      style={{
        border: `1px solid ${palette.border}`,
        backgroundColor: palette.background,
        color: palette.color,
      }}
    >
      <div className="n3-error-banner__body">
        {errors?.length ? (
          <ul>
            {errors.map((entry, index) => {
              const value = typeof entry === "string" ? entry : entry?.message ?? entry?.detail ?? "";
              return (
                <li key={index}>{value}</li>
              );
            })}
          </ul>
        ) : null}
        {children}
      </div>
      {action ? <div className="n3-error-banner__action">{action}</div> : null}
      {typeof onDismiss === "function" ? (
        <button type="button" className="n3-error-banner__dismiss" onClick={onDismiss} aria-label="Dismiss">
          ×
        </button>
      ) : null}
    </section>
  );
}
