export interface User {
  id: number;
  username: string;
  role: string;
  status: string;
  created_at: string;
  approved_at: string | null;
  last_login_at: string | null;
}

export interface CatalogModel {
  id: string;
  provider: string;
  label: string | null;
  note?: string | null;
  pricing?: { input_usd_per_1m: number | null; output_usd_per_1m: number | null } | null;
}

export interface OcrUsage {
  input_tokens: number | null;
  output_tokens: number | null;
  estimated_cost_usd: number | null;
}

export interface OcrResult {
  text: string;
  lines: { text: string; confidence: number; box: number[] | null }[];
  line_count: number;
  image: {
    width: number;
    height: number;
    original_width: number;
    original_height: number;
    format: string | null;
    content_type: string | null;
  };
  model: { provider: string; id: string; label: string | null };
  usage: OcrUsage | null;
  processing_time_ms: number;
}

const TOKEN_KEY = "ocr.session.token";
const USER_KEY = "ocr.session.user";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function getStoredUser(): User | null {
  const raw = localStorage.getItem(USER_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as User;
  } catch {
    return null;
  }
}

export function setSession(token: string, user: User): void {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, JSON.stringify(user));
}

export function clearSession(): void {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request(
  path: string,
  init: RequestInit = {},
  options: { auth?: boolean } = { auth: true },
): Promise<Response> {
  const headers = new Headers(init.headers);
  if (options.auth !== false) {
    const token = getToken();
    if (token) headers.set("Authorization", `Bearer ${token}`);
  }
  const response = await fetch(path, { ...init, headers });
  if (response.status === 401 && options.auth !== false) {
    // Session is gone or invalid; force re-login.
    clearSession();
  }
  return response;
}

async function parseError(response: Response): Promise<ApiError> {
  let detail = `HTTP ${response.status}`;
  try {
    const body = await response.json();
    if (typeof body.detail === "string") detail = body.detail;
    else if (Array.isArray(body.detail) && body.detail[0]?.msg) detail = body.detail[0].msg;
  } catch {
    // keep generic detail
  }
  return new ApiError(response.status, detail);
}

export async function login(username: string, password: string): Promise<{ user: User }> {
  const response = await request("/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  }, { auth: false });
  if (!response.ok) throw await parseError(response);
  const body = (await response.json()) as { token: string; user: User };
  setSession(body.token, body.user);
  return { user: body.user };
}

export async function register(username: string, password: string): Promise<string> {
  const response = await request("/auth/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  }, { auth: false });
  if (!response.ok) throw await parseError(response);
  const body = (await response.json()) as { detail: string };
  return body.detail;
}

export async function fetchMe(): Promise<{ kind: string; user: User | null }> {
  const response = await request("/auth/me");
  if (!response.ok) throw await parseError(response);
  return (await response.json()) as { kind: string; user: User | null };
}

export async function fetchModels(): Promise<CatalogModel[]> {
  const response = await request("/models");
  if (!response.ok) throw await parseError(response);
  const body = (await response.json()) as { models: CatalogModel[] };
  return body.models;
}

export async function runOcr(
  file: Blob,
  filename: string,
  model: string,
  minConfidence: number,
): Promise<OcrResult> {
  const form = new FormData();
  form.append("file", file, filename);
  const params = new URLSearchParams();
  if (model) params.set("model", model);
  if (minConfidence > 0) params.set("min_confidence", String(minConfidence));
  const query = params.toString();
  const response = await request(`/ocr${query ? `?${query}` : ""}`, {
    method: "POST",
    body: form,
  });
  if (!response.ok) throw await parseError(response);
  return (await response.json()) as OcrResult;
}

export async function fetchUsers(): Promise<User[]> {
  const response = await request("/admin/users");
  if (!response.ok) throw await parseError(response);
  return ((await response.json()) as { users: User[] }).users;
}

export async function setUserStatus(userId: number, action: "approve" | "disable"): Promise<User> {
  const response = await request(`/admin/users/${userId}/${action}`, { method: "POST" });
  if (!response.ok) throw await parseError(response);
  return ((await response.json()) as { user: User }).user;
}

export async function deleteUser(userId: number): Promise<void> {
  const response = await request(`/admin/users/${userId}`, { method: "DELETE" });
  if (!response.ok) throw await parseError(response);
}
