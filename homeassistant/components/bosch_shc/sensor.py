"""Platform for sensor integration."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, override

from boschshcpy import (
    CommunicationQualityMixin,
    HumidityLevelMixin,
    PowerMeterMixin,
    SHCSmartPlugCompact,
    SHCThermostat,
    SHCTwinguard,
    SHCWallThermostat,
    TemperatureLevelMixin,
)
from boschshcpy.device import SHCDevice

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.const import (
    UnitOfEnergy,
    UnitOfPower,
    UnitOfRatio,
    UnitOfTemperature,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import StateType

from . import BoschConfigEntry
from .entity import SHCEntity

# Contravariant so a description typed against a capability is accepted
# wherever one typed against a concrete device class is expected; mypy
# cannot infer this through dataclass fields (hence no PEP 695 syntax).
_DeviceT_contra = TypeVar("_DeviceT_contra", bound=SHCDevice, contravariant=True)


@dataclass(frozen=True, kw_only=True)
class SHCSensorEntityDescription(SensorEntityDescription, Generic[_DeviceT_contra]):  # noqa: UP046
    """Describes a SHC sensor.

    Typed against the narrowest capability the callables read, so one
    description serves every device class exposing that capability.
    """

    value_fn: Callable[[_DeviceT_contra], StateType]
    attributes_fn: Callable[[_DeviceT_contra], dict[str, Any]] | None = None


TEMPERATURE_SENSOR = "temperature"
HUMIDITY_SENSOR = "humidity"
VALVE_TAPPET_SENSOR = "valvetappet"
PURITY_SENSOR = "purity"
AIR_QUALITY_SENSOR = "airquality"
TEMPERATURE_RATING_SENSOR = "temperature_rating"
HUMIDITY_RATING_SENSOR = "humidity_rating"
PURITY_RATING_SENSOR = "purity_rating"
POWER_SENSOR = "power"
ENERGY_SENSOR = "energy"
COMMUNICATION_QUALITY_SENSOR = "communication_quality"

_TEMPERATURE_DESCRIPTION: SHCSensorEntityDescription[TemperatureLevelMixin] = (
    SHCSensorEntityDescription(
        key=TEMPERATURE_SENSOR,
        device_class=SensorDeviceClass.TEMPERATURE,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        value_fn=lambda device: device.temperature,
    )
)
_HUMIDITY_DESCRIPTION: SHCSensorEntityDescription[HumidityLevelMixin] = (
    SHCSensorEntityDescription(
        key=HUMIDITY_SENSOR,
        device_class=SensorDeviceClass.HUMIDITY,
        native_unit_of_measurement=UnitOfRatio.PERCENTAGE,
        value_fn=lambda device: device.humidity,
    )
)
_VALVE_TAPPET_DESCRIPTION: SHCSensorEntityDescription[SHCThermostat] = (
    SHCSensorEntityDescription(
        key=VALVE_TAPPET_SENSOR,
        translation_key=VALVE_TAPPET_SENSOR,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=UnitOfRatio.PERCENTAGE,
        value_fn=lambda device: device.position,
        attributes_fn=lambda device: {
            "valve_tappet_state": device.valvestate.name,
        },
    )
)
_PURITY_DESCRIPTION: SHCSensorEntityDescription[SHCTwinguard] = (
    SHCSensorEntityDescription(
        key=PURITY_SENSOR,
        translation_key=PURITY_SENSOR,
        native_unit_of_measurement=UnitOfRatio.PARTS_PER_MILLION,
        value_fn=lambda device: device.purity,
    )
)
_AIR_QUALITY_DESCRIPTION: SHCSensorEntityDescription[SHCTwinguard] = (
    SHCSensorEntityDescription(
        key=AIR_QUALITY_SENSOR,
        translation_key="air_quality",
        value_fn=lambda device: device.combined_rating.name,
        attributes_fn=lambda device: {
            "rating_description": device.description,
        },
    )
)
_TEMPERATURE_RATING_DESCRIPTION: SHCSensorEntityDescription[SHCTwinguard] = (
    SHCSensorEntityDescription(
        key=TEMPERATURE_RATING_SENSOR,
        translation_key=TEMPERATURE_RATING_SENSOR,
        value_fn=lambda device: device.temperature_rating.name,
    )
)
_HUMIDITY_RATING_DESCRIPTION: SHCSensorEntityDescription[SHCTwinguard] = (
    SHCSensorEntityDescription(
        key=HUMIDITY_RATING_SENSOR,
        translation_key=HUMIDITY_RATING_SENSOR,
        value_fn=lambda device: device.humidity_rating.name,
    )
)
_PURITY_RATING_DESCRIPTION: SHCSensorEntityDescription[SHCTwinguard] = (
    SHCSensorEntityDescription(
        key=PURITY_RATING_SENSOR,
        translation_key=PURITY_RATING_SENSOR,
        value_fn=lambda device: device.purity_rating.name,
    )
)
_POWER_DESCRIPTION: SHCSensorEntityDescription[PowerMeterMixin] = (
    SHCSensorEntityDescription(
        key=POWER_SENSOR,
        device_class=SensorDeviceClass.POWER,
        native_unit_of_measurement=UnitOfPower.WATT,
        value_fn=lambda device: device.powerconsumption,
    )
)
_ENERGY_DESCRIPTION: SHCSensorEntityDescription[PowerMeterMixin] = (
    SHCSensorEntityDescription(
        key=ENERGY_SENSOR,
        device_class=SensorDeviceClass.ENERGY,
        state_class=SensorStateClass.TOTAL_INCREASING,
        native_unit_of_measurement=UnitOfEnergy.KILO_WATT_HOUR,
        value_fn=lambda device: device.energyconsumption / 1000.0,
    )
)
_COMMUNICATION_QUALITY_DESCRIPTION: SHCSensorEntityDescription[
    CommunicationQualityMixin
] = SHCSensorEntityDescription(
    key=COMMUNICATION_QUALITY_SENSOR,
    translation_key=COMMUNICATION_QUALITY_SENSOR,
    value_fn=lambda device: device.communicationquality.name,
)

_THERMOSTAT_DESCRIPTIONS: tuple[SHCSensorEntityDescription[SHCThermostat], ...] = (
    _TEMPERATURE_DESCRIPTION,
    _VALVE_TAPPET_DESCRIPTION,
)
_WALLTHERMOSTAT_DESCRIPTIONS: tuple[
    SHCSensorEntityDescription[SHCWallThermostat], ...
] = (
    _TEMPERATURE_DESCRIPTION,
    _HUMIDITY_DESCRIPTION,
)
_TWINGUARD_DESCRIPTIONS: tuple[SHCSensorEntityDescription[SHCTwinguard], ...] = (
    _TEMPERATURE_DESCRIPTION,
    _HUMIDITY_DESCRIPTION,
    _PURITY_DESCRIPTION,
    _AIR_QUALITY_DESCRIPTION,
    _TEMPERATURE_RATING_DESCRIPTION,
    _HUMIDITY_RATING_DESCRIPTION,
    _PURITY_RATING_DESCRIPTION,
)
_POWER_METER_DESCRIPTIONS: tuple[SHCSensorEntityDescription[PowerMeterMixin], ...] = (
    _POWER_DESCRIPTION,
    _ENERGY_DESCRIPTION,
)
_SMART_PLUG_COMPACT_DESCRIPTIONS: tuple[
    SHCSensorEntityDescription[SHCSmartPlugCompact], ...
] = (
    _POWER_DESCRIPTION,
    _ENERGY_DESCRIPTION,
    _COMMUNICATION_QUALITY_DESCRIPTION,
)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: BoschConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the SHC sensor platform."""
    session = config_entry.runtime_data

    entities: list[SensorEntity] = [
        SHCSensor(
            device,
            description,
            session.unique_id,
            config_entry.entry_id,
        )
        for device in session.device_helper.thermostats
        for description in _THERMOSTAT_DESCRIPTIONS
    ]

    entities.extend(
        SHCSensor(
            device,
            description,
            session.unique_id,
            config_entry.entry_id,
        )
        for device in session.device_helper.wallthermostats
        for description in _WALLTHERMOSTAT_DESCRIPTIONS
    )

    entities.extend(
        SHCSensor(
            device,
            description,
            session.unique_id,
            config_entry.entry_id,
        )
        for device in session.device_helper.twinguards
        for description in _TWINGUARD_DESCRIPTIONS
    )

    entities.extend(
        SHCSensor(
            device,
            description,
            session.unique_id,
            config_entry.entry_id,
        )
        for device in (
            *session.device_helper.smart_plugs,
            *session.device_helper.light_switches_bsm,
        )
        for description in _POWER_METER_DESCRIPTIONS
    )

    entities.extend(
        SHCSensor(
            device,
            description,
            session.unique_id,
            config_entry.entry_id,
        )
        for device in session.device_helper.smart_plugs_compact
        for description in _SMART_PLUG_COMPACT_DESCRIPTIONS
    )

    async_add_entities(entities)


class SHCSensor[_DeviceT: SHCDevice](SHCEntity, SensorEntity):
    """Representation of a SHC sensor."""

    entity_description: SHCSensorEntityDescription[_DeviceT]

    def __init__(
        self,
        device: _DeviceT,
        entity_description: SHCSensorEntityDescription[_DeviceT],
        parent_id: str,
        entry_id: str,
    ) -> None:
        """Initialize sensor."""
        super().__init__(device, parent_id, entry_id)
        self._device: _DeviceT = device
        self.entity_description = entity_description
        self._attr_unique_id = f"{device.serial}_{entity_description.key}"

    @property
    @override
    def native_value(self) -> StateType:
        """Return the state of the sensor."""
        return self.entity_description.value_fn(self._device)

    @property
    @override
    def extra_state_attributes(self) -> dict[str, Any] | None:
        """Return the state attributes."""
        if self.entity_description.attributes_fn is not None:
            return self.entity_description.attributes_fn(self._device)
        return None
