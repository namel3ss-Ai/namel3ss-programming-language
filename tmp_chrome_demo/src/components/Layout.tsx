import { NavLink } from "../lib/navigation";
import { Link, useLocation } from "react-router-dom";
import type { PropsWithChildren } from "react";

interface LayoutProps {
  title: string;
  description?: string | null;
  navLinks: readonly NavLink[];
}

export default function Layout({ title, description, navLinks, children }: PropsWithChildren<LayoutProps>) {
  const location = useLocation();

  return (
    <div className="n3-app">
      <header style={{ padding: "1.25rem clamp(1rem, 4vw, 4rem)" }}>
        <h1 style={{ marginBottom: "0.25rem" }}>{title}</h1>
        {description ? <p style={{ marginTop: 0, color: "var(--text-muted, #475569)" }}>{description}</p> : null}
        <nav className="n3-nav">
          {navLinks.map((link) => (
            <Link key={link.path} to={link.path} className={location.pathname === link.path ? "active" : undefined}>
              {link.label}
            </Link>
          ))}
        </nav>
      </header>
      <main className="n3-main">{children}</main>
    </div>
  );
}
