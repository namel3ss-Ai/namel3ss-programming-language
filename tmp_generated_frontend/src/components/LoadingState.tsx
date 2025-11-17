import type { CSSProperties } from "react";
import { useI18n } from "../lib/i18n";

export function PageLoader({ style }: { style?: CSSProperties }) {
  const { t } = useI18n();
  return (
    <div role="status" aria-live="polite" className="n3-loader" style={style}>
      <span className="n3-loader__spinner" aria-hidden="true" />
      <span>{t("loading.generic")}</span>
    </div>
  );
}

export function SectionSkeleton({ lines = 3 }: { lines?: number }) {
  const placeholders = Array.from({ length: Math.max(1, lines) });
  return (
    <div className="n3-skeleton" aria-hidden="true">
      {placeholders.map((_, index) => (
        <div key={index} className="n3-skeleton__line" style={{ width: `${80 - index * 7}%` }} />
      ))}
    </div>
  );
}

export function InlineSpinner() {
  return <span className="n3-inline-spinner" aria-hidden="true" />;
}
