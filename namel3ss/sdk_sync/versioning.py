"""
Schema versioning and migration system.

Provides:
    1. Version compatibility checking
    2. Migration generation (upgrade/downgrade)
    3. Breaking change detection
    4. Migration validation

Example:
    Check compatibility:
    ```python
    checker = CompatibilityChecker()
    is_compatible = checker.check_compatibility(v1_model, v2_model)
    if not is_compatible:
        migration = MigrationGenerator().generate(v1_model, v2_model)
    ```
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .ir import (
    IRModel,
    IRField,
    IRType,
    SchemaVersion,
    SchemaMigration,
)
from .errors import MigrationError, VersionMismatchError


class ChangeType(str, Enum):
    """Types of schema changes."""

    FIELD_ADDED = "field_added"
    FIELD_REMOVED = "field_removed"
    FIELD_RENAMED = "field_renamed"
    FIELD_TYPE_CHANGED = "field_type_changed"
    FIELD_REQUIRED_CHANGED = "field_required_changed"
    CONSTRAINT_ADDED = "constraint_added"
    CONSTRAINT_REMOVED = "constraint_removed"
    CONSTRAINT_CHANGED = "constraint_changed"


@dataclass
class SchemaChange:
    """Represents a single schema change."""

    type: ChangeType
    field_name: Optional[str]
    old_value: Any
    new_value: Any
    breaking: bool
    description: str


class CompatibilityChecker:
    """
    Checks compatibility between schema versions.
    
    Determines if changes are:
    - Backward compatible (safe)
    - Forward compatible (safe)
    - Breaking (incompatible)
    """

    def check_compatibility(
        self, old_model: IRModel, new_model: IRModel
    ) -> Tuple[bool, List[SchemaChange]]:
        """
        Check if new model is compatible with old model.
        
        Args:
            old_model: Original model
            new_model: New model version
        
        Returns:
            (is_compatible, list_of_changes)
        """
        changes = self._detect_changes(old_model, new_model)
        is_compatible = all(not c.breaking for c in changes)
        return is_compatible, changes

    def _detect_changes(
        self, old_model: IRModel, new_model: IRModel
    ) -> List[SchemaChange]:
        """Detect all changes between models."""
        changes = []

        # Build field maps
        old_fields = {f.name: f for f in old_model.fields}
        new_fields = {f.name: f for f in new_model.fields}

        # Detect removed fields
        for name, old_field in old_fields.items():
            if name not in new_fields:
                changes.append(
                    SchemaChange(
                        type=ChangeType.FIELD_REMOVED,
                        field_name=name,
                        old_value=old_field,
                        new_value=None,
                        breaking=old_field.required,  # Breaking if required
                        description=f"Field '{name}' removed",
                    )
                )

        # Detect added fields
        for name, new_field in new_fields.items():
            if name not in old_fields:
                changes.append(
                    SchemaChange(
                        type=ChangeType.FIELD_ADDED,
                        field_name=name,
                        old_value=None,
                        new_value=new_field,
                        breaking=False,  # Adding fields is safe
                        description=f"Field '{name}' added",
                    )
                )

        # Detect changed fields
        for name in set(old_fields.keys()) & set(new_fields.keys()):
            old_field = old_fields[name]
            new_field = new_fields[name]
            field_changes = self._detect_field_changes(old_field, new_field)
            changes.extend(field_changes)

        return changes

    def _detect_field_changes(
        self, old_field: IRField, new_field: IRField
    ) -> List[SchemaChange]:
        """Detect changes in a single field."""
        changes = []

        # Type change
        if old_field.type != new_field.type:
            changes.append(
                SchemaChange(
                    type=ChangeType.FIELD_TYPE_CHANGED,
                    field_name=old_field.name,
                    old_value=old_field.type,
                    new_value=new_field.type,
                    breaking=True,  # Type changes are breaking
                    description=f"Field '{old_field.name}' type changed from {old_field.type} to {new_field.type}",
                )
            )

        # Required changed
        if old_field.required != new_field.required:
            breaking = new_field.required  # Making required is breaking
            changes.append(
                SchemaChange(
                    type=ChangeType.FIELD_REQUIRED_CHANGED,
                    field_name=old_field.name,
                    old_value=old_field.required,
                    new_value=new_field.required,
                    breaking=breaking,
                    description=f"Field '{old_field.name}' required changed: {old_field.required} -> {new_field.required}",
                )
            )

        # Constraints changed
        constraint_changes = self._detect_constraint_changes(old_field, new_field)
        changes.extend(constraint_changes)

        return changes

    def _detect_constraint_changes(
        self, old_field: IRField, new_field: IRField
    ) -> List[SchemaChange]:
        """Detect constraint changes."""
        changes = []

        old_constraints = set(old_field.constraints.keys())
        new_constraints = set(new_field.constraints.keys())

        # Added constraints
        for key in new_constraints - old_constraints:
            changes.append(
                SchemaChange(
                    type=ChangeType.CONSTRAINT_ADDED,
                    field_name=old_field.name,
                    old_value=None,
                    new_value=new_field.constraints[key],
                    breaking=True,  # Adding constraints is breaking
                    description=f"Constraint '{key}' added to field '{old_field.name}'",
                )
            )

        # Removed constraints
        for key in old_constraints - new_constraints:
            changes.append(
                SchemaChange(
                    type=ChangeType.CONSTRAINT_REMOVED,
                    field_name=old_field.name,
                    old_value=old_field.constraints[key],
                    new_value=None,
                    breaking=False,  # Removing constraints is safe
                    description=f"Constraint '{key}' removed from field '{old_field.name}'",
                )
            )

        # Changed constraints
        for key in old_constraints & new_constraints:
            if old_field.constraints[key] != new_field.constraints[key]:
                # Determine if breaking
                breaking = self._is_constraint_change_breaking(
                    key, old_field.constraints[key], new_field.constraints[key]
                )
                changes.append(
                    SchemaChange(
                        type=ChangeType.CONSTRAINT_CHANGED,
                        field_name=old_field.name,
                        old_value=old_field.constraints[key],
                        new_value=new_field.constraints[key],
                        breaking=breaking,
                        description=f"Constraint '{key}' changed on field '{old_field.name}'",
                    )
                )

        return changes

    def _is_constraint_change_breaking(
        self, constraint_name: str, old_value: Any, new_value: Any
    ) -> bool:
        """Determine if constraint change is breaking."""
        # More restrictive = breaking
        if constraint_name in {"minLength", "min_length", "minimum"}:
            return new_value > old_value
        elif constraint_name in {"maxLength", "max_length", "maximum"}:
            return new_value < old_value
        else:
            return True  # Assume breaking if unsure


class MigrationGenerator:
    """
    Generates schema migrations with upgrade/downgrade code.
    
    Migrations are explicit, versioned, and type-safe.
    """

    def generate(
        self, old_model: IRModel, new_model: IRModel
    ) -> SchemaMigration:
        """
        Generate migration between model versions.
        
        Args:
            old_model: Original model
            new_model: Target model
        
        Returns:
            Complete migration specification
        """
        checker = CompatibilityChecker()
        is_compatible, changes = checker.check_compatibility(old_model, new_model)

        # Generate migration code
        upgrade_code = self._generate_upgrade_code(changes)
        downgrade_code = self._generate_downgrade_code(changes)

        return SchemaMigration(
            schema_name=old_model.name,
            from_version=old_model.version,
            to_version=new_model.version,
            description=self._generate_description(changes),
            changes=[
                {
                    "type": c.type,
                    "field": c.field_name,
                    "old": str(c.old_value) if c.old_value else None,
                    "new": str(c.new_value) if c.new_value else None,
                    "breaking": c.breaking,
                }
                for c in changes
            ],
            breaking=not is_compatible,
            upgrade_code=upgrade_code,
            downgrade_code=downgrade_code,
        )

    def _generate_upgrade_code(self, changes: List[SchemaChange]) -> str:
        """Generate Python code for upgrade migration."""
        lines = []
        lines.append("def upgrade(data: Dict[str, Any]) -> Dict[str, Any]:")
        lines.append('    """Upgrade data from old schema to new schema."""')
        lines.append("    migrated = data.copy()")
        lines.append("")

        for change in changes:
            if change.type == ChangeType.FIELD_REMOVED:
                lines.append(f"    # Remove field: {change.field_name}")
                lines.append(f"    migrated.pop('{change.field_name}', None)")
            elif change.type == ChangeType.FIELD_ADDED:
                if change.new_value and change.new_value.default is not None:
                    lines.append(f"    # Add field: {change.field_name}")
                    lines.append(
                        f"    migrated.setdefault('{change.field_name}', {repr(change.new_value.default)})"
                    )
            elif change.type == ChangeType.FIELD_TYPE_CHANGED:
                lines.append(f"    # TODO: Convert type for field: {change.field_name}")
                lines.append(f"    # Old type: {change.old_value}")
                lines.append(f"    # New type: {change.new_value}")

        lines.append("")
        lines.append("    return migrated")

        return "\n".join(lines)

    def _generate_downgrade_code(self, changes: List[SchemaChange]) -> str:
        """Generate Python code for downgrade migration."""
        lines = []
        lines.append("def downgrade(data: Dict[str, Any]) -> Dict[str, Any]:")
        lines.append('    """Downgrade data from new schema to old schema."""')
        lines.append("    migrated = data.copy()")
        lines.append("")

        # Reverse the changes
        for change in reversed(changes):
            if change.type == ChangeType.FIELD_ADDED:
                lines.append(f"    # Remove added field: {change.field_name}")
                lines.append(f"    migrated.pop('{change.field_name}', None)")
            elif change.type == ChangeType.FIELD_REMOVED:
                if change.old_value and change.old_value.default is not None:
                    lines.append(f"    # Restore removed field: {change.field_name}")
                    lines.append(
                        f"    migrated.setdefault('{change.field_name}', {repr(change.old_value.default)})"
                    )

        lines.append("")
        lines.append("    return migrated")

        return "\n".join(lines)

    def _generate_description(self, changes: List[SchemaChange]) -> str:
        """Generate human-readable migration description."""
        if not changes:
            return "No changes"

        descriptions = [c.description for c in changes]
        return "; ".join(descriptions[:5])  # First 5 changes


class VersionManager:
    """
    Manages schema versions and migrations.
    
    Provides:
    - Version tracking
    - Migration path finding
    - Migration execution
    - Rollback support
    """

    def __init__(self):
        """Initialize version manager."""
        self.migrations: Dict[str, List[SchemaMigration]] = {}

    def register_migration(self, migration: SchemaMigration) -> None:
        """Register a migration."""
        if migration.schema_name not in self.migrations:
            self.migrations[migration.schema_name] = []
        self.migrations[migration.schema_name].append(migration)

    def find_migration_path(
        self,
        schema_name: str,
        from_version: SchemaVersion,
        to_version: SchemaVersion,
    ) -> List[SchemaMigration]:
        """
        Find migration path between versions.
        
        Args:
            schema_name: Schema name
            from_version: Start version
            to_version: Target version
        
        Returns:
            List of migrations to apply
        
        Raises:
            MigrationError: If no path found
        """
        if schema_name not in self.migrations:
            raise MigrationError(
                f"No migrations found for schema: {schema_name}",
                from_version=str(from_version),
                to_version=str(to_version),
                schema_name=schema_name,
            )

        # Simple linear search for now
        # Production version would use graph algorithm
        schema_migrations = sorted(
            self.migrations[schema_name],
            key=lambda m: m.from_version,
        )

        path = []
        current_version = from_version

        for migration in schema_migrations:
            if migration.from_version == current_version:
                path.append(migration)
                current_version = migration.to_version
                if current_version == to_version:
                    return path

        raise MigrationError(
            f"No migration path from {from_version} to {to_version}",
            from_version=str(from_version),
            to_version=str(to_version),
            schema_name=schema_name,
        )

    def execute_migration(
        self, migration: SchemaMigration, data: Dict[str, Any], direction: str = "upgrade"
    ) -> Dict[str, Any]:
        """
        Execute migration on data.
        
        Args:
            migration: Migration to execute
            data: Data to migrate
            direction: "upgrade" or "downgrade"
        
        Returns:
            Migrated data
        """
        if direction == "upgrade":
            code = migration.upgrade_code
        else:
            code = migration.downgrade_code

        if not code:
            return data

        # Execute migration code
        # In production, this would be sandboxed
        namespace = {"Dict": Dict, "Any": Any}
        exec(code, namespace)
        migrate_func = namespace.get(direction)

        if not migrate_func:
            raise MigrationError(
                f"Migration code missing {direction} function",
                from_version=str(migration.from_version),
                to_version=str(migration.to_version),
                schema_name=migration.schema_name,
            )

        return migrate_func(data)
