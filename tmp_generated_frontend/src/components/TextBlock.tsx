import type { CSSProperties } from "react";
import type { TextWidgetConfig } from "../lib/n3Client";

interface TextBlockProps {
  widget: TextWidgetConfig;
}

function normaliseStyles(styles: Record<string, string> | undefined): Record<string, string> {
  const result: Record<string, string> = {};
  if (!styles) {
    return result;
  }

  const sizeScale: Record<string, string> = {
    small: "0.875rem",
    medium: "1rem",
    large: "1.35rem",
    "x-large": "1.75rem",
    "xx-large": "2rem",
  };

  Object.entries(styles).forEach(([rawKey, rawValue]) => {
    const key = rawKey.toLowerCase();
    const value = rawValue;
    if (key === "align") {
      result.textAlign = value;
      return;
    }
    if (key === "weight") {
      const weight = value.toLowerCase();
      if (weight === "bold") {
        result.fontWeight = "700";
      } else if (weight === "light") {
        result.fontWeight = "300";
      } else if (weight === "normal") {
        result.fontWeight = "400";
      } else {
        result.fontWeight = value;
      }
      return;
    }
    if (key === "size") {
      const size = sizeScale[value.toLowerCase()] ?? value;
      result.fontSize = size;
      return;
    }
    const parts = rawKey.split(/[-_\s]+/).filter(Boolean);
    if (!parts.length) {
      return;
    }
    const camel = parts[0] + parts.slice(1).map((segment) => segment.charAt(0).toUpperCase() + segment.slice(1)).join("");
    result[camel] = value;
  });
  return result;
}

export default function TextBlock({ widget }: TextBlockProps) {
  return (
    <section className="n3-widget" style={normaliseStyles(widget.styles) as CSSProperties}>
      <p style={{ margin: 0 }}>{widget.text}</p>
    </section>
  );
}
