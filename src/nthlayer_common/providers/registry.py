from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Callable
import builtins

ProviderFactory = Callable[..., Any]


@dataclass(frozen=True)
class ProviderSpec:
    """Metadata describing a registered provider."""

    name: str
    factory: ProviderFactory
    version: str | None = None
    description: str | None = None


class ProviderRegistry:
    """Simple in-memory registry for NthLayer providers."""

    def __init__(self) -> None:
        self._providers: dict[str, ProviderSpec] = {}

    def register(
        self,
        name: str,
        factory: ProviderFactory,
        *,
        version: str | None = None,
        description: str | None = None,
    ) -> None:
        if not name:
            raise ValueError("Provider name is required")
        spec = ProviderSpec(
            name=name,
            factory=factory,
            version=version,
            description=description,
        )
        self._providers[name] = spec

    def create(self, name: str, **kwargs: Any) -> Any:
        spec = self._providers.get(name)
        if spec is None:
            raise KeyError(f"Provider '{name}' is not registered")
        return spec.factory(**kwargs)

    def list(self) -> builtins.list[ProviderSpec]:
        return list(self._providers.values())


# This singleton is populated by side-effect imports in nthlayer.providers.__init__.
# Built-in providers (grafana, pagerduty) register themselves when their modules
# are imported. Consumers outside nthlayer must either import nthlayer.providers
# first or register providers explicitly before calling create_provider().
provider_registry = ProviderRegistry()


def register_provider(
    name: str,
    factory: ProviderFactory,
    *,
    version: str | None = None,
    description: str | None = None,
) -> None:
    provider_registry.register(name, factory, version=version, description=description)


def create_provider(name: str, **kwargs: Any) -> Any:
    return provider_registry.create(name, **kwargs)


def list_providers() -> list[ProviderSpec]:
    return provider_registry.list()
