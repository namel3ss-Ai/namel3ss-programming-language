export interface NavLink {
          label: string;
          path: string;
          requiresAuth?: boolean;
          allowedRoles?: readonly string[];
          hideWhenAuthenticated?: boolean;
          onlyWhenAuthenticated?: boolean;
        }

        export const NAV_LINKS: NavLink[] = [
  {
    "label": "Home",
    "path": "/",
    "requiresAuth": false,
    "allowedRoles": [],
    "hideWhenAuthenticated": false,
    "onlyWhenAuthenticated": false
  },
  {
    "label": "Feedback",
    "path": "/feedback",
    "requiresAuth": false,
    "allowedRoles": [],
    "hideWhenAuthenticated": false,
    "onlyWhenAuthenticated": false
  },
  {
    "label": "Admin",
    "path": "/admin",
    "requiresAuth": false,
    "allowedRoles": [],
    "hideWhenAuthenticated": false,
    "onlyWhenAuthenticated": false
  }
] as const;
