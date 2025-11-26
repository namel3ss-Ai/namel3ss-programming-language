import { createContext, PropsWithChildren, useCallback, useContext, useMemo, useState } from "react";

interface ToastContextValue {
  show: (message: string, timeoutMs?: number) => void;
}

const ToastContext = createContext<ToastContextValue>({
  show: () => undefined,
});

export function ToastProvider({ children }: PropsWithChildren) {
  const [message, setMessage] = useState<string | null>(null);
  const [timer, setTimer] = useState<number | undefined>(undefined);

  const show = useCallback((nextMessage: string, timeoutMs = 2800) => {
    setMessage(nextMessage);
    if (timer) {
      window.clearTimeout(timer);
    }
    const id = window.setTimeout(() => setMessage(null), timeoutMs);
    setTimer(id);
  }, [timer]);

  const value = useMemo<ToastContextValue>(() => ({ show }), [show]);

  return (
    <ToastContext.Provider value={value}>
      {children}
      {message ? <div className="n3-toast" role="status">{message}</div> : null}
    </ToastContext.Provider>
  );
}

export function useToast() {
  return useContext(ToastContext);
}
