import { useState, useEffect, useCallback, useRef } from "react";
import { useNavigate } from "react-router-dom";

export interface CommandSource {
  type: "routes" | "actions" | "custom";
  filter?: string;
  custom_items?: Array<{ label: string; action?: string }>;
}

export interface CommandPaletteProps {
  shortcut?: string;
  sources: CommandSource[];
  placeholder?: string;
  max_results?: number;
  available_routes?: Array<{ label: string; path: string }>;
  available_actions?: Array<{ label: string; id: string }>;
}

interface Command {
  id: string;
  label: string;
  type: "route" | "action";
  target?: string;  // route path or action ID
}

export default function CommandPalette({
  shortcut = "ctrl+k",
  sources,
  placeholder = "Search commands...",
  max_results = 10,
  available_routes = [],
  available_actions = [],
}: CommandPaletteProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  // Build command list from sources
  const allCommands = useMemo((): Command[] => {
    const commands: Command[] = [];

    sources.forEach(source => {
      if (source.type === "routes") {
        available_routes.forEach(route => {
          commands.push({
            id: `route:${route.path}`,
            label: route.label,
            type: "route",
            target: route.path,
          });
        });
      } else if (source.type === "actions") {
        available_actions.forEach(action => {
          commands.push({
            id: `action:${action.id}`,
            label: action.label,
            type: "action",
            target: action.id,
          });
        });
      } else if (source.type === "custom" && source.custom_items) {
        source.custom_items.forEach(item => {
          commands.push({
            id: `custom:${item.label}`,
            label: item.label,
            type: "action",
            target: item.action,
          });
        });
      }
    });

    return commands;
  }, [sources, available_routes, available_actions]);

  // Filter commands based on query
  const filteredCommands = useMemo(() => {
    if (!query) return allCommands.slice(0, max_results);

    const lowerQuery = query.toLowerCase();
    return allCommands
      .filter(cmd => cmd.label.toLowerCase().includes(lowerQuery))
      .slice(0, max_results);
  }, [query, allCommands, max_results]);

  // Handle keyboard shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Parse shortcut (e.g., "ctrl+k", "cmd+k")
      const parts = shortcut.toLowerCase().split('+');
      const hasCtrl = parts.includes('ctrl') || parts.includes('cmd');
      const key = parts[parts.length - 1];

      if (hasCtrl && (e.ctrlKey || e.metaKey) && e.key.toLowerCase() === key) {
        e.preventDefault();
        setIsOpen(prev => !prev);
      }

      // Close on Escape
      if (e.key === 'Escape' && isOpen) {
        setIsOpen(false);
        setQuery("");
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [shortcut, isOpen]);

  // Focus input when opened
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  // Handle command navigation with arrow keys
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex(prev => 
        prev < filteredCommands.length - 1 ? prev + 1 : prev
      );
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex(prev => prev > 0 ? prev - 1 : prev);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      const command = filteredCommands[selectedIndex];
      if (command) executeCommand(command);
    }
  }, [filteredCommands, selectedIndex]);

  const executeCommand = (command: Command) => {
    if (command.type === "route" && command.target) {
      navigate(command.target);
    } else if (command.type === "action" && command.target) {
      window.dispatchEvent(new CustomEvent('namel3ss:action', {
        detail: { actionId: command.target }
      }));
    }
    setIsOpen(false);
    setQuery("");
    setSelectedIndex(0);
  };

  if (!isOpen) return null;

  return (
    <>
      <div className="command-palette-overlay" onClick={() => setIsOpen(false)} />
      <div className="command-palette" role="dialog" aria-label="Command palette">
        <div className="command-palette-input-wrapper">
          <input
            ref={inputRef}
            type="text"
            className="command-palette-input"
            placeholder={placeholder}
            value={query}
            onChange={e => {
              setQuery(e.target.value);
              setSelectedIndex(0);
            }}
            onKeyDown={handleKeyDown}
            aria-label="Search commands"
            aria-autocomplete="list"
            aria-controls="command-list"
          />
        </div>
        <ul id="command-list" className="command-palette-results" role="listbox">
          {filteredCommands.length === 0 ? (
            <li className="command-palette-no-results">No commands found</li>
          ) : (
            filteredCommands.map((command, index) => (
              <li
                key={command.id}
                className={`command-palette-item ${index === selectedIndex ? 'selected' : ''}`}
                onClick={() => executeCommand(command)}
                role="option"
                aria-selected={index === selectedIndex}
              >
                <span className="command-label">{command.label}</span>
                <span className="command-type">{command.type}</span>
              </li>
            ))
          )}
        </ul>
      </div>
    </>
  );
}
