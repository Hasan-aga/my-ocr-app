from pydantic import BaseModel, Field


class OcrLine(BaseModel):
    text: str
    confidence: float = Field(ge=0, le=1)
    box: list[float] | None = None


class ImageInfo(BaseModel):
    width: int
    height: int
    original_width: int
    original_height: int
    format: str | None = None
    content_type: str | None = None


class ModelInfo(BaseModel):
    provider: str
    id: str
    label: str | None = None
    tier: str | None = None
    detection: str | None = None
    recognition: str | None = None


class Pricing(BaseModel):
    input_usd_per_1m: float | None = Field(default=None, ge=0)
    output_usd_per_1m: float | None = Field(default=None, ge=0)


class CatalogModel(BaseModel):
    id: str
    provider: str
    label: str | None = None
    note: str | None = None
    pricing: Pricing | None = None


class ModelsResponse(BaseModel):
    models: list[CatalogModel]


class UsageInfo(BaseModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)


class OcrResponse(BaseModel):
    text: str
    lines: list[OcrLine]
    line_count: int
    image: ImageInfo
    model: ModelInfo
    usage: UsageInfo | None = None
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    ready: bool
    model: ModelInfo


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=8, max_length=256)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=8, max_length=256)


class UserOut(BaseModel):
    id: int
    username: str
    role: str
    status: str
    created_at: str
    approved_at: str | None = None
    last_login_at: str | None = None


class RegisterResponse(BaseModel):
    status: str
    detail: str


class LoginResponse(BaseModel):
    token: str
    expires_at: int
    user: UserOut


class MeResponse(BaseModel):
    kind: str
    user: UserOut | None = None


class UserListResponse(BaseModel):
    users: list[UserOut]


class UserStatusResponse(BaseModel):
    user: UserOut