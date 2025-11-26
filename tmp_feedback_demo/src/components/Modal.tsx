import { useState, useEffect, ReactNode } from "react";
import { X } from "lucide-react";

export interface ModalAction {
  label: string;
  action?: string;
  variant?: "default" | "primary" | "destructive" | "ghost" | "link";
  close?: boolean;
}

export interface ModalProps {
  id: string;
  title: string;
  description?: string;
  content: ReactNode[];
  actions: ModalAction[];
  size?: "sm" | "md" | "lg" | "xl" | "full";
  dismissible?: boolean;
  trigger?: string;
  isOpen?: boolean;
  onClose?: () => void;
}

export default function Modal({
  id,
  title,
  description,
  content,
  actions,
  size = "md",
  dismissible = true,
  trigger,
  isOpen: controlledIsOpen,
  onClose,
}: ModalProps) {
  const [internalIsOpen, setInternalIsOpen] = useState(false);
  const isControlled = controlledIsOpen !== undefined;
  const isOpen = isControlled ? controlledIsOpen : internalIsOpen;

  useEffect(() => {
    if (!trigger) return;

    const handleAction = (event: CustomEvent) => {
      if (event.detail.actionId === trigger) {
        if (isControlled && onClose) {
          onClose(); // Let parent handle
        } else {
          setInternalIsOpen(true);
        }
      }
    };

    window.addEventListener('namel3ss:action' as any, handleAction);
    return () => window.removeEventListener('namel3ss:action' as any, handleAction);
  }, [trigger, isControlled, onClose]);

  const handleClose = () => {
    if (isControlled && onClose) {
      onClose();
    } else {
      setInternalIsOpen(false);
    }
  };

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (dismissible && e.target === e.currentTarget) {
      handleClose();
    }
  };

  const handleActionClick = (action: ModalAction) => {
    if (action.action) {
      window.dispatchEvent(new CustomEvent('namel3ss:action', {
        detail: { actionId: action.action }
      }));
    }
    if (action.close !== false) {
      handleClose();
    }
  };

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && dismissible && isOpen) {
        handleClose();
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = '';
    };
  }, [isOpen, dismissible]);

  if (!isOpen) return null;

  const sizeClasses = {
    sm: 'max-w-sm',
    md: 'max-w-md',
    lg: 'max-w-lg',
    xl: 'max-w-xl',
    full: 'max-w-full mx-4'
  };

  const variantClasses = {
    default: 'bg-white text-gray-900 border-gray-200 hover:bg-gray-50',
    primary: 'bg-blue-600 text-white hover:bg-blue-700',
    destructive: 'bg-red-600 text-white hover:bg-red-700',
    ghost: 'bg-transparent hover:bg-gray-100',
    link: 'text-blue-600 underline-offset-4 hover:underline'
  };

  return (
    <div
      className="modal-overlay fixed inset-0 bg-black/50 flex items-center justify-center z-50"
      onClick={handleBackdropClick}
      role="dialog"
      aria-modal="true"
      aria-labelledby={`modal-${id}-title`}
      aria-describedby={description ? `modal-${id}-desc` : undefined}
    >
      <div className={`modal-content bg-white rounded-lg shadow-xl ${sizeClasses[size]} w-full`}>
        <div className="modal-header flex items-start justify-between p-6 border-b">
          <div>
            <h2 id={`modal-${id}-title`} className="text-xl font-semibold text-gray-900">
              {title}
            </h2>
            {description && (
              <p id={`modal-${id}-desc`} className="text-sm text-gray-500 mt-1">
                {description}
              </p>
            )}
          </div>
          {dismissible && (
            <button
              onClick={handleClose}
              className="text-gray-400 hover:text-gray-600 transition-colors"
              aria-label="Close modal"
            >
              <X size={20} />
            </button>
          )}
        </div>

        <div className="modal-body p-6">
          {content}
        </div>

        {actions.length > 0 && (
          <div className="modal-footer flex gap-2 justify-end p-6 border-t bg-gray-50">
            {actions.map((action, index) => (
              <button
                key={index}
                onClick={() => handleActionClick(action)}
                className={`px-4 py-2 rounded-md font-medium transition-colors ${
                  variantClasses[action.variant || 'default']
                }`}
              >
                {action.label}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
