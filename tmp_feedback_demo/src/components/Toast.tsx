import { useState, useEffect } from "react";
import { X, CheckCircle, XCircle, AlertCircle, Info } from "lucide-react";

export interface ToastProps {
  id: string;
  title: string;
  description?: string;
  variant?: "default" | "success" | "error" | "warning" | "info";
  duration?: number;
  action_label?: string;
  action?: string;
  position?: "top" | "top-right" | "top-left" | "bottom" | "bottom-right" | "bottom-left";
  trigger?: string;
  isVisible?: boolean;
  onDismiss?: () => void;
}

export default function Toast({
  id,
  title,
  description,
  variant = "default",
  duration = 3000,
  action_label,
  action,
  position = "top-right",
  trigger,
  isVisible: controlledIsVisible,
  onDismiss,
}: ToastProps) {
  const [internalIsVisible, setInternalIsVisible] = useState(false);
  const isControlled = controlledIsVisible !== undefined;
  const isVisible = isControlled ? controlledIsVisible : internalIsVisible;

  useEffect(() => {
    if (!trigger) return;

    const handleAction = (event: CustomEvent) => {
      if (event.detail.actionId === trigger) {
        if (isControlled && onDismiss) {
          // Parent controls visibility
        } else {
          setInternalIsVisible(true);
        }
      }
    };

    window.addEventListener('namel3ss:action' as any, handleAction);
    return () => window.removeEventListener('namel3ss:action' as any, handleAction);
  }, [trigger, isControlled, onDismiss]);

  useEffect(() => {
    if (isVisible && duration > 0) {
      const timer = setTimeout(() => {
        handleDismiss();
      }, duration);
      return () => clearTimeout(timer);
    }
  }, [isVisible, duration]);

  const handleDismiss = () => {
    if (isControlled && onDismiss) {
      onDismiss();
    } else {
      setInternalIsVisible(false);
    }
  };

  const handleActionClick = () => {
    if (action) {
      window.dispatchEvent(new CustomEvent('namel3ss:action', {
        detail: { actionId: action }
      }));
    }
    handleDismiss();
  };

  if (!isVisible) return null;

  const variantStyles = {
    default: {
      bg: 'bg-white border-gray-200',
      text: 'text-gray-900',
      icon: null
    },
    success: {
      bg: 'bg-green-50 border-green-200',
      text: 'text-green-900',
      icon: <CheckCircle size={20} className="text-green-600" />
    },
    error: {
      bg: 'bg-red-50 border-red-200',
      text: 'text-red-900',
      icon: <XCircle size={20} className="text-red-600" />
    },
    warning: {
      bg: 'bg-yellow-50 border-yellow-200',
      text: 'text-yellow-900',
      icon: <AlertCircle size={20} className="text-yellow-600" />
    },
    info: {
      bg: 'bg-blue-50 border-blue-200',
      text: 'text-blue-900',
      icon: <Info size={20} className="text-blue-600" />
    }
  };

  const positionClasses = {
    'top': 'top-4 left-1/2 -translate-x-1/2',
    'top-right': 'top-4 right-4',
    'top-left': 'top-4 left-4',
    'bottom': 'bottom-4 left-1/2 -translate-x-1/2',
    'bottom-right': 'bottom-4 right-4',
    'bottom-left': 'bottom-4 left-4'
  };

  const style = variantStyles[variant];

  return (
    <div
      className={`toast fixed ${positionClasses[position]} z-50 w-96 max-w-full animate-slide-in`}
      role="alert"
      aria-live="polite"
      aria-atomic="true"
    >
      <div className={`${style.bg} border rounded-lg shadow-lg p-4`}>
        <div className="flex gap-3">
          {style.icon && <div className="flex-shrink-0">{style.icon}</div>}
          <div className="flex-1 min-w-0">
            <h3 className={`font-semibold ${style.text}`}>{title}</h3>
            {description && (
              <p className={`text-sm mt-1 ${style.text} opacity-90`}>
                {description}
              </p>
            )}
            {action_label && (
              <button
                onClick={handleActionClick}
                className={`text-sm font-medium mt-2 ${style.text} underline hover:no-underline`}
              >
                {action_label}
              </button>
            )}
          </div>
          <button
            onClick={handleDismiss}
            className={`flex-shrink-0 ${style.text} opacity-50 hover:opacity-100 transition-opacity`}
            aria-label="Dismiss notification"
          >
            <X size={16} />
          </button>
        </div>
      </div>
    </div>
  );
}
