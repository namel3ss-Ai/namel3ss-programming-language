"""
Utilities for generating TypeScript/React code with design token support.

This module provides helper functions that integrate with the central design token
mapping layer to generate consistent Tailwind CSS classes in React components.
"""

import textwrap
from pathlib import Path
from typing import Optional

from .utils import write_file


def write_design_tokens_util(lib_dir: Path) -> None:
    """
    Generate designTokens.ts utility for mapping design tokens to Tailwind classes.
    
    This TypeScript module mirrors the Python mapping layer and provides runtime
    token-to-class conversion for React components.
    """
    content = textwrap.dedent(
        """
        /**
         * Design Token Utilities
         * 
         * Converts design tokens to Tailwind CSS classes.
         * Mirrors the Python mapping layer in /namel3ss/codegen/frontend/design_token_mapping.py
         */

        import { useState, useEffect } from 'react';

        export type VariantType = 'elevated' | 'outlined' | 'ghost' | 'subtle';
        export type ToneType = 'neutral' | 'primary' | 'success' | 'warning' | 'danger';
        export type DensityType = 'comfortable' | 'compact';
        export type SizeType = 'xs' | 'sm' | 'md' | 'lg' | 'xl';
        export type ThemeType = 'light' | 'dark' | 'system';
        export type ColorSchemeType = 'blue' | 'green' | 'violet' | 'rose' | 'orange' | 'teal' | 'indigo' | 'slate';

        export interface ComponentDesignTokens {
          variant?: VariantType;
          tone?: ToneType;
          density?: DensityType;
          size?: SizeType;
        }

        export interface AppLevelDesignTokens {
          theme?: ThemeType;
          color_scheme?: ColorSchemeType;
        }

        /**
         * Map button design tokens to Tailwind CSS classes
         */
        export function mapButtonClasses(
          variant?: VariantType,
          tone?: ToneType,
          size?: SizeType
        ): string {
          const classes: string[] = ['transition-colors', 'font-semibold', 'rounded-md'];

          // Variant + Tone combinations
          if (variant === 'elevated') {
            if (tone === 'primary') classes.push('bg-blue-600 hover:bg-blue-700 text-white shadow-md');
            else if (tone === 'success') classes.push('bg-green-600 hover:bg-green-700 text-white shadow-md');
            else if (tone === 'warning') classes.push('bg-amber-600 hover:bg-amber-700 text-white shadow-md');
            else if (tone === 'danger') classes.push('bg-red-600 hover:bg-red-700 text-white shadow-md');
            else classes.push('bg-gray-700 hover:bg-gray-800 text-white shadow-md');
          } else if (variant === 'outlined') {
            if (tone === 'primary') classes.push('border-2 border-blue-600 text-blue-600 hover:bg-blue-50');
            else if (tone === 'success') classes.push('border-2 border-green-600 text-green-600 hover:bg-green-50');
            else if (tone === 'warning') classes.push('border-2 border-amber-600 text-amber-600 hover:bg-amber-50');
            else if (tone === 'danger') classes.push('border-2 border-red-600 text-red-600 hover:bg-red-50');
            else classes.push('border-2 border-gray-600 text-gray-600 hover:bg-gray-50');
          } else if (variant === 'ghost') {
            if (tone === 'primary') classes.push('text-blue-600 hover:bg-blue-50');
            else if (tone === 'success') classes.push('text-green-600 hover:bg-green-50');
            else if (tone === 'warning') classes.push('text-amber-600 hover:bg-amber-50');
            else if (tone === 'danger') classes.push('text-red-600 hover:bg-red-50');
            else classes.push('text-gray-600 hover:bg-gray-100');
          } else if (variant === 'subtle') {
            if (tone === 'primary') classes.push('bg-blue-100 text-blue-700 hover:bg-blue-200');
            else if (tone === 'success') classes.push('bg-green-100 text-green-700 hover:bg-green-200');
            else if (tone === 'warning') classes.push('bg-amber-100 text-amber-700 hover:bg-amber-200');
            else if (tone === 'danger') classes.push('bg-red-100 text-red-700 hover:bg-red-200');
            else classes.push('bg-gray-100 text-gray-700 hover:bg-gray-200');
          }

          // Size classes
          if (size === 'xs') classes.push('px-2 py-1 text-xs');
          else if (size === 'sm') classes.push('px-3 py-1.5 text-sm');
          else if (size === 'lg') classes.push('px-6 py-3 text-lg');
          else if (size === 'xl') classes.push('px-8 py-4 text-xl');
          else classes.push('px-4 py-2 text-base'); // md is default

          return classes.join(' ');
        }

        /**
         * Map input design tokens to Tailwind CSS classes
         */
        export function mapInputClasses(
          variant?: VariantType,
          tone?: ToneType,
          size?: SizeType
        ): string {
          const classes: string[] = ['rounded-md', 'transition-colors', 'w-full'];

          // Variant styles
          if (variant === 'outlined' || !variant) {
            classes.push('border-2 border-gray-300 focus:border-blue-500 focus:ring-2 focus:ring-blue-200');
          } else if (variant === 'ghost') {
            classes.push('border-0 bg-transparent hover:bg-gray-50 focus:bg-gray-50');
          } else if (variant === 'subtle') {
            classes.push('border-0 bg-gray-100 focus:bg-gray-200');
          }

          // Size classes
          if (size === 'xs') classes.push('px-2 py-1 text-xs');
          else if (size === 'sm') classes.push('px-3 py-1.5 text-sm');
          else if (size === 'lg') classes.push('px-5 py-3 text-lg');
          else if (size === 'xl') classes.push('px-6 py-4 text-xl');
          else classes.push('px-4 py-2 text-base'); // md is default

          return classes.join(' ');
        }

        /**
         * Map card design tokens to Tailwind CSS classes
         */
        export function mapCardClasses(
          variant?: VariantType,
          tone?: ToneType,
          density?: DensityType
        ): string {
          const classes: string[] = ['rounded-lg'];

          // Variant styles
          if (variant === 'elevated' || !variant) {
            classes.push('bg-white shadow-lg border border-gray-200');
          } else if (variant === 'outlined') {
            classes.push('bg-white border-2 border-gray-300');
          } else if (variant === 'ghost') {
            classes.push('bg-transparent');
          } else if (variant === 'subtle') {
            classes.push('bg-gray-50 border border-gray-200');
          }

          // Tone accent (border/highlight color)
          if (tone === 'primary') classes.push('border-l-4 border-l-blue-600');
          else if (tone === 'success') classes.push('border-l-4 border-l-green-600');
          else if (tone === 'warning') classes.push('border-l-4 border-l-amber-600');
          else if (tone === 'danger') classes.push('border-l-4 border-l-red-600');

          // Density (padding)
          if (density === 'compact') classes.push('p-3');
          else classes.push('p-6'); // comfortable is default

          return classes.join(' ');
        }

        /**
         * Map form container design tokens to Tailwind CSS classes
         */
        export function mapFormClasses(
          variant?: VariantType,
          tone?: ToneType,
          size?: SizeType
        ): string {
          const classes: string[] = ['rounded-lg'];

          // Variant styles for form container
          if (variant === 'elevated') {
            classes.push('bg-white shadow-lg border border-gray-200 p-6');
          } else if (variant === 'outlined') {
            classes.push('bg-white border-2 border-gray-300 p-6');
          } else if (variant === 'ghost') {
            classes.push('bg-transparent p-0');
          } else if (variant === 'subtle') {
            classes.push('bg-gray-50 border border-gray-200 p-6');
          } else {
            // Default: outlined
            classes.push('bg-white border-2 border-gray-300 p-6');
          }

          return classes.join(' ');
        }

        /**
         * Map table design tokens to Tailwind CSS classes
         */
        export function mapTableClasses(
          variant?: VariantType,
          tone?: ToneType,
          size?: SizeType,
          density?: DensityType
        ): string {
          const classes: string[] = ['w-full', 'border-collapse', 'text-left'];

          // Variant styles for table container
          if (variant === 'elevated' || !variant) {
            classes.push('bg-white shadow-sm rounded-lg overflow-hidden');
          } else if (variant === 'outlined') {
            classes.push('border border-gray-200 rounded-lg');
          } else if (variant === 'ghost') {
            classes.push('border-0');
          } else if (variant === 'subtle') {
            classes.push('bg-gray-50 border border-gray-100');
          }

          // Tone (header background)
          if (tone === 'neutral' || !tone) {
            classes.push('[&_thead]:bg-gray-100');
          } else if (tone === 'primary') {
            classes.push('[&_thead]:bg-blue-50');
          } else if (tone === 'success') {
            classes.push('[&_thead]:bg-green-50');
          } else if (tone === 'warning') {
            classes.push('[&_thead]:bg-amber-50');
          } else if (tone === 'danger') {
            classes.push('[&_thead]:bg-red-50');
          }

          // Size (cell padding and text)
          if (size === 'xs') {
            classes.push('[&_th]:px-2 [&_th]:py-1 [&_td]:px-2 [&_td]:py-1 [&_th]:text-xs [&_td]:text-xs');
          } else if (size === 'sm') {
            classes.push('[&_th]:px-3 [&_th]:py-1.5 [&_td]:px-3 [&_td]:py-1.5 [&_th]:text-sm [&_td]:text-sm');
          } else if (size === 'lg') {
            classes.push('[&_th]:px-6 [&_th]:py-3 [&_td]:px-6 [&_td]:py-3 [&_th]:text-lg [&_td]:text-lg');
          } else if (size === 'xl') {
            classes.push('[&_th]:px-8 [&_th]:py-4 [&_td]:px-8 [&_td]:py-4 [&_th]:text-xl [&_td]:text-xl');
          } else {
            // md is default
            classes.push('[&_th]:px-4 [&_th]:py-2 [&_td]:px-4 [&_td]:py-2 [&_th]:text-base [&_td]:text-base');
          }

          // Density (row height)
          if (density === 'compact') {
            classes.push('[&_tbody_tr]:h-8');
          } else {
            // comfortable is default
            classes.push('[&_tbody_tr]:h-12');
          }

          // Row styling
          classes.push('[&_tbody_tr]:border-b [&_tbody_tr]:border-gray-200');
          classes.push('[&_tbody_tr:hover]:bg-gray-50');

          return classes.join(' ');
        }

        /**
         * Map badge design tokens to Tailwind CSS classes
         */
        export function mapBadgeClasses(
          variant?: VariantType,
          tone?: ToneType,
          size?: SizeType
        ): string {
          const classes: string[] = ['inline-flex', 'items-center', 'font-medium', 'rounded-full'];

          // Variant styles
          if (variant === 'elevated') {
            classes.push('shadow-sm');
          } else if (variant === 'outlined') {
            classes.push('border-2');
          } else if (variant === 'ghost') {
            classes.push('border-0', 'bg-transparent');
          } else if (variant === 'subtle') {
            classes.push('border');
          } else {
            // Default: elevated
            classes.push('shadow-sm');
          }

          // Tone colors
          if (tone === 'primary') {
            if (variant === 'outlined') {
              classes.push('border-blue-500', 'text-blue-700', 'bg-white');
            } else if (variant === 'ghost') {
              classes.push('text-blue-600');
            } else {
              classes.push('bg-blue-100', 'text-blue-800');
            }
          } else if (tone === 'success') {
            if (variant === 'outlined') {
              classes.push('border-green-500', 'text-green-700', 'bg-white');
            } else if (variant === 'ghost') {
              classes.push('text-green-600');
            } else {
              classes.push('bg-green-100', 'text-green-800');
            }
          } else if (tone === 'warning') {
            if (variant === 'outlined') {
              classes.push('border-amber-500', 'text-amber-700', 'bg-white');
            } else if (variant === 'ghost') {
              classes.push('text-amber-600');
            } else {
              classes.push('bg-amber-100', 'text-amber-800');
            }
          } else if (tone === 'danger') {
            if (variant === 'outlined') {
              classes.push('border-red-500', 'text-red-700', 'bg-white');
            } else if (variant === 'ghost') {
              classes.push('text-red-600');
            } else {
              classes.push('bg-red-100', 'text-red-800');
            }
          } else {
            // neutral
            if (variant === 'outlined') {
              classes.push('border-gray-300', 'text-gray-700', 'bg-white');
            } else if (variant === 'ghost') {
              classes.push('text-gray-600');
            } else {
              classes.push('bg-gray-100', 'text-gray-800');
            }
          }

          // Size
          if (size === 'xs') {
            classes.push('px-2', 'py-0.5', 'text-xs');
          } else if (size === 'sm') {
            classes.push('px-2.5', 'py-0.5', 'text-sm');
          } else if (size === 'lg') {
            classes.push('px-4', 'py-1', 'text-base');
          } else if (size === 'xl') {
            classes.push('px-5', 'py-1.5', 'text-lg');
          } else {
            // md is default
            classes.push('px-3', 'py-1', 'text-sm');
          }

          return classes.join(' ');
        }

        /**
         * Map alert/notification design tokens to Tailwind CSS classes
         */
        export function mapAlertClasses(
          variant?: VariantType,
          tone?: ToneType,
          size?: SizeType
        ): string {
          const classes: string[] = ['rounded-lg', 'p-4'];

          // Variant styles
          if (variant === 'elevated') {
            classes.push('shadow-md');
          } else if (variant === 'outlined') {
            classes.push('border-2');
          } else if (variant === 'ghost') {
            classes.push('border-0', 'bg-transparent');
          } else if (variant === 'subtle') {
            classes.push('border');
          } else {
            // Default: subtle
            classes.push('border');
          }

          // Tone colors
          if (tone === 'primary') {
            if (variant === 'outlined') {
              classes.push('bg-blue-50', 'border-blue-500', 'text-blue-900');
            } else if (variant === 'ghost') {
              classes.push('text-blue-700');
            } else {
              classes.push('bg-blue-50', 'border-blue-200', 'text-blue-900');
            }
          } else if (tone === 'success') {
            if (variant === 'outlined') {
              classes.push('bg-green-50', 'border-green-500', 'text-green-900');
            } else if (variant === 'ghost') {
              classes.push('text-green-700');
            } else {
              classes.push('bg-green-50', 'border-green-200', 'text-green-900');
            }
          } else if (tone === 'warning') {
            if (variant === 'outlined') {
              classes.push('bg-amber-50', 'border-amber-500', 'text-amber-900');
            } else if (variant === 'ghost') {
              classes.push('text-amber-700');
            } else {
              classes.push('bg-amber-50', 'border-amber-200', 'text-amber-900');
            }
          } else if (tone === 'danger') {
            if (variant === 'outlined') {
              classes.push('bg-red-50', 'border-red-500', 'text-red-900');
            } else if (variant === 'ghost') {
              classes.push('text-red-700');
            } else {
              classes.push('bg-red-50', 'border-red-200', 'text-red-900');
            }
          } else {
            // neutral
            if (variant === 'outlined') {
              classes.push('bg-gray-50', 'border-gray-500', 'text-gray-900');
            } else if (variant === 'ghost') {
              classes.push('text-gray-700');
            } else {
              classes.push('bg-gray-50', 'border-gray-200', 'text-gray-900');
            }
          }

          // Size
          if (size === 'xs') {
            classes.push('text-xs', 'p-2');
          } else if (size === 'sm') {
            classes.push('text-sm', 'p-3');
          } else if (size === 'lg') {
            classes.push('text-lg', 'p-5');
          } else if (size === 'xl') {
            classes.push('text-xl', 'p-6');
          } else {
            // md is default (already set p-4 above)
            classes.push('text-base');
          }

          return classes.join(' ');
        }

        /**
         * Map density token to spacing/height classes
         */
        export function mapDensityClasses(density?: DensityType): string {
          const classes: string[] = [];

          if (density === 'compact') {
            classes.push('space-y-1', 'py-1');
          } else if (density === 'comfortable') {
            classes.push('space-y-3', 'py-2');
          } else {
            // default to comfortable
            classes.push('space-y-3', 'py-2');
          }

          return classes.join(' ');
        }

        /**
         * Get theme class name for app container
         */
        export function getThemeClassName(theme?: ThemeType): string {
          if (theme === 'dark') return 'dark';
          if (theme === 'light') return '';
          
          // For system theme, detect OS preference
          if (theme === 'system') {
            if (typeof window !== 'undefined' && window.matchMedia) {
              const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
              return prefersDark ? 'dark' : '';
            }
          }
          
          return ''; // default to light
        }

        /**
         * React hook to detect and respond to system theme changes
         */
        export function useSystemTheme(theme?: ThemeType): string {
          if (typeof window === 'undefined') return '';
          if (theme !== 'system') return getThemeClassName(theme);
          
          const [isDark, setIsDark] = useState(() => {
            return window.matchMedia('(prefers-color-scheme: dark)').matches;
          });
          
          useEffect(() => {
            if (theme !== 'system') return;
            
            const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
            const handler = (e: MediaQueryListEvent) => setIsDark(e.matches);
            
            mediaQuery.addEventListener('change', handler);
            return () => mediaQuery.removeEventListener('change', handler);
          }, [theme]);
          
          return isDark ? 'dark' : '';
        }

        /**
         * Get color scheme CSS variable override
         */
        export function getColorSchemeStyles(colorScheme?: ColorSchemeType): Record<string, string> {
          if (!colorScheme) return {};

          const schemes = {
            blue: { '--primary': '#2563eb', '--primary-hover': '#1d4ed8' },
            green: { '--primary': '#16a34a', '--primary-hover': '#15803d' },
            violet: { '--primary': '#7c3aed', '--primary-hover': '#6d28d9' },
            rose: { '--primary': '#e11d48', '--primary-hover': '#be123c' },
            orange: { '--primary': '#ea580c', '--primary-hover': '#c2410c' },
            teal: { '--primary': '#0d9488', '--primary-hover': '#0f766e' },
            indigo: { '--primary': '#4f46e5', '--primary-hover': '#4338ca' },
            slate: { '--primary': '#475569', '--primary-hover': '#334155' },
          };

          return schemes[colorScheme] || {};
        }
        """
    ).strip() + "\n"
    
    write_file(lib_dir / "designTokens.ts", content)
