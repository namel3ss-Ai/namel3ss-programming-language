import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { fetchResource } from "./n3Client";

export type AuthStatus = "loading" | "authenticated" | "unauthenticated";

export interface AuthUser extends Record<string, unknown> {
  id?: string | number;
  email?: string;
  name?: string;
}

export interface AuthSession {
  token?: string | null;
  user?: AuthUser | null;
  roles?: readonly string[] | null;
  authenticated?: boolean;
  expiresAt?: string | null;
}

export interface AuthState {
  status: AuthStatus;
  user: AuthUser | null;
  token: string | null;
  roles: string[];
  expiresAt?: string | null;
  error: string | null;
}

export interface AuthContextValue extends AuthState {
  ready: boolean;
  login: (session: AuthSession) => void;
  logout: () => void;
  refresh: () => Promise<void>;
  setSession: (session: Partial<AuthSession>) => void;
  hasRole: (role: string) => boolean;
  hasAnyRole: (roles: readonly string[]) => boolean;
}

export interface AuthProviderProps {
  children: ReactNode;
  storageKey?: string;
  sessionEndpoint?: string | null;
  initialSession?: AuthSession | null;
}

const STORAGE_KEY = "namel3ss.auth.v1";

const defaultState: AuthState = {
  status: "loading",
  user: null,
  token: null,
  roles: [],
  expiresAt: undefined,
  error: null,
};

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

const isBrowser = typeof window !== "undefined" && typeof window.localStorage !== "undefined";

function normaliseRoles(roles: unknown): string[] {
  if (!roles) {
    return [];
  }
  if (Array.isArray(roles)) {
    return roles.map((entry) => String(entry));
  }
  if (typeof roles === "string" && roles.trim()) {
    return roles.split(/[,\s]+/).filter(Boolean);
  }
  return [];
}

function normaliseSession(session: AuthSession | null | undefined): AuthState {
  if (!session) {
    return { ...defaultState, status: "unauthenticated" };
  }
  const roles = normaliseRoles(session.roles);
  const token = typeof session.token === "string" ? session.token : null;
  const user = session.user && typeof session.user === "object" ? session.user : null;
  const expiresAt = typeof session.expiresAt === "string" ? session.expiresAt : undefined;
  const authenticated = session.authenticated ?? Boolean(token || user);
  return {
    status: authenticated ? "authenticated" : "unauthenticated",
    user,
    token,
    roles,
    expiresAt,
    error: null,
  };
}

function loadStoredSession(storageKey: string): AuthState | null {
  if (!isBrowser) {
    return null;
  }
  try {
    const raw = window.localStorage.getItem(storageKey);
    if (!raw) {
      return null;
    }
    const parsed = JSON.parse(raw) as AuthSession;
    return normaliseSession(parsed);
  } catch (error) {
    console.warn("Failed to read auth session from storage", error);
    return null;
  }
}

function persistSession(storageKey: string, state: AuthState) {
  if (!isBrowser) {
    return;
  }
  try {
    if (state.status === "unauthenticated") {
      window.localStorage.removeItem(storageKey);
      return;
    }
    const payload: AuthSession = {
      token: state.token,
      user: state.user,
      roles: state.roles,
      authenticated: state.status === "authenticated",
      expiresAt: state.expiresAt ?? null,
    };
    window.localStorage.setItem(storageKey, JSON.stringify(payload));
  } catch (error) {
    console.warn("Failed to persist auth session", error);
  }
}

export class UnauthenticatedError extends Error {
  constructor(message = "User is not authenticated") {
    super(message);
    this.name = "UnauthenticatedError";
  }
}

export function AuthProvider({
  children,
  storageKey = STORAGE_KEY,
  sessionEndpoint = "/api/auth/session",
  initialSession = null,
}: AuthProviderProps) {
  const [state, setState] = useState<AuthState>(() => {
    if (initialSession) {
      return normaliseSession(initialSession);
    }
    const stored = loadStoredSession(storageKey);
    if (stored) {
      return stored;
    }
    return { ...defaultState, status: sessionEndpoint ? "loading" : "unauthenticated" };
  });
  const readyRef = useRef(state.status !== "loading");
  const [ready, setReady] = useState(readyRef.current);
  const refreshingRef = useRef(false);

  useEffect(() => {
    if (state.status !== "loading" && !readyRef.current) {
      readyRef.current = true;
      setReady(true);
    }
  }, [state.status]);

  useEffect(() => {
    if (!readyRef.current && !sessionEndpoint) {
      readyRef.current = true;
      setReady(true);
      setState((prev) => ({ ...prev, status: prev.status === "loading" ? "unauthenticated" : prev.status }));
    }
  }, [sessionEndpoint]);

  useEffect(() => {
    persistSession(storageKey, state);
  }, [state, storageKey]);

  const applySession = useCallback((next: AuthSession | null, options?: { error?: string | null }) => {
    const base = normaliseSession(next);
    const errorMessage = options?.error ?? null;
    setState({ ...base, error: errorMessage });
    readyRef.current = true;
    setReady(true);
  }, []);

  const refresh = useCallback(async () => {
    if (!sessionEndpoint || refreshingRef.current) {
      return;
    }
    refreshingRef.current = true;
    try {
      const session = await fetchResource<AuthSession>(sessionEndpoint, {
        credentials: "include",
      });
      applySession(session ?? null, { error: null });
    } catch (error) {
      console.warn("Failed to refresh auth session", error);
      const message = error instanceof Error && error.message ? error.message : "Unable to verify session.";
      applySession(null, { error: message });
    } finally {
      refreshingRef.current = false;
    }
  }, [sessionEndpoint, applySession]);

  useEffect(() => {
    if (state.status === "loading" && sessionEndpoint) {
      refresh();
    }
  }, [state.status, refresh, sessionEndpoint]);

  const login = useCallback((session: AuthSession) => {
    applySession({ ...session, authenticated: true }, { error: null });
  }, [applySession]);

  const logout = useCallback(() => {
    applySession(null, { error: null });
  }, [applySession]);

  const setSession = useCallback((session: Partial<AuthSession>) => {
    setState((prev) => {
      const merged: AuthSession = {
        token: session.token ?? prev.token,
        user: (session.user ?? prev.user) as AuthUser | null,
        roles: session.roles ?? prev.roles,
        authenticated: session.authenticated ?? (prev.status === "authenticated"),
        expiresAt: session.expiresAt ?? prev.expiresAt ?? null,
      };
      readyRef.current = true;
      setReady(true);
      const base = normaliseSession(merged);
      return { ...base, error: null };
    });
  }, []);

  const hasRole = useCallback((role: string) => {
    return state.roles.includes(role);
  }, [state.roles]);

  const hasAnyRole = useCallback((roles: readonly string[]) => {
    if (!roles || roles.length === 0) {
      return true;
    }
    return roles.some((role) => state.roles.includes(role));
  }, [state.roles]);

  const value = useMemo<AuthContextValue>(() => ({
    ...state,
    ready,
    login,
    logout,
    refresh,
    setSession,
    hasRole,
    hasAnyRole,
  }), [state, ready, login, logout, refresh, setSession, hasRole, hasAnyRole]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}

export function useAuthenticatedUser(): AuthUser | null {
  const { user, status } = useAuth();
  return status === "authenticated" ? user : null;
}

export function assertAuthenticated(): void {
  const { status } = useAuth();
  if (status !== "authenticated") {
    throw new UnauthenticatedError();
  }
}
