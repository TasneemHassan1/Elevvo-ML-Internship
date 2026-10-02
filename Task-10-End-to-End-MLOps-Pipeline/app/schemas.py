from pydantic import BaseModel, ConfigDict, Field


class MachineInput(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True
    )

    Type: str = Field(
        ...,
        description="Machine type: L, M, or H"
    )

    air_temperature: float = Field(
        ...,
        alias="Air temperature [K]"
    )

    process_temperature: float = Field(
        ...,
        alias="Process temperature [K]"
    )

    rotational_speed: float = Field(
        ...,
        alias="Rotational speed [rpm]"
    )

    torque: float = Field(
        ...,
        alias="Torque [Nm]"
    )

    tool_wear: float = Field(
        ...,
        alias="Tool wear [min]"
    )