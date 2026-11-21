import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { ApiError, fetchModels, runOcr } from "../api";
import type { CatalogModel, OcrResult } from "../api";
import { useAuth } from "../auth";

interface SelectedImage {
  file: File;
  name: string;
  size: number;
  url: string;
}

export default function OcrPage() {
  const { user, signOut } = useAuth();
  const [models, setModels] = useState<CatalogModel[]>([]);
  const [model, setModel] = useState("");
  const [minConfidence, setMinConfidence] = useState(0);
  const [image, setImage] = useState<SelectedImage | null>(null);
  const [status, setStatus] = useState<{ kind: "info" | "error" | "success"; text: string } | null>(
    null,
  );
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState<OcrResult | null>(null);

  // One input for the camera, one for the library; both native and reliable.
  const cameraInputRef = useRef<HTMLInputElement | null>(null);
  const libraryInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    fetchModels()
      .then((list) => {
        setModels(list);
        if (list.length > 0) setModel(list[0].id);
      })
      .catch(() => setStatus({ kind: "error", text: "Could not load the model list." }));
  }, []);

  function onFileChosen(file: File | null | undefined) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setStatus({ kind: "error", text: "Please choose an image file." });
      return;
    }
    if (file.size === 0) {
      setStatus({ kind: "error", text: "That file is empty." });
      return;
    }
    const url = URL.createObjectURL(file);
    setImage((previous) => {
      if (previous) URL.revokeObjectURL(previous.url);
      return { file, name: file.name || "photo.jpg", size: file.size, url };
    });
    setResult(null);
    setStatus(null);
  }

  async function process() {
    if (!image) {
      setStatus({ kind: "error", text: "Take a photo or choose a file first." });
      return;
    }
    setProcessing(true);
    setStatus({ kind: "info", text: "Processing…" });
    setResult(null);
    try {
      const ocr = await runOcr(image.file, image.name, model, minConfidence);
      setResult(ocr);
      setStatus({ kind: "success", text: "Done." });
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        setStatus({ kind: "error", text: "Session expired. Sign in again." });
      } else {
        setStatus({ kind: "error", text: err instanceof Error ? err.message : "OCR failed." });
      }
    } finally {
      setProcessing(false);
    }
  }

  const formatSize = (bytes: number) =>
    bytes >= 1024 * 1024 ? `${(bytes / (1024 * 1024)).toFixed(1)} MB` : `${Math.round(bytes / 1024)} KB`;

  return (
    <main className="page">
      <header className="topbar">
        <h1>My OCR App</h1>
        <div className="topbar-actions">
          <span className="muted">{user?.username}</span>
          {user?.role === "admin" && <Link to="/admin">Users</Link>}
          <button className="btn" onClick={signOut}>Sign out</button>
        </div>
      </header>

      <section className="card">
        <h2>Image</h2>
        <div className="upload-options">
          {/* capture="environment" opens the rear camera directly to snap a
              photo on phones; on desktop it falls back to the file picker. */}
          <button className="btn btn-primary" onClick={() => cameraInputRef.current?.click()}>
            📸 Take photo
          </button>
          <button className="btn" onClick={() => libraryInputRef.current?.click()}>
            🖼 Choose from library
          </button>
          <input
            ref={cameraInputRef}
            type="file"
            accept="image/*"
            capture="environment"
            hidden
            onChange={(e) => {
              onFileChosen(e.target.files?.[0]);
              e.target.value = ""; // allow re-picking the same file
            }}
          />
          <input
            ref={libraryInputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => {
              onFileChosen(e.target.files?.[0]);
              e.target.value = "";
            }}
          />
        </div>

        {image && (
          <>
            <img className="preview" src={image.url} alt="Selected" />
            <p className="muted">
              {image.name} · {formatSize(image.size)}
            </p>
          </>
        )}
      </section>

      <section className="card">
        <h2>Process</h2>
        <div className="form-row">
          <label htmlFor="model">Model</label>
          <select id="model" value={model} onChange={(e) => setModel(e.target.value)}>
            {models.map((entry) => (
              <option key={entry.id} value={entry.id}>
                {entry.label ?? entry.id}
              </option>
            ))}
          </select>
        </div>
        <div className="form-row">
          <label htmlFor="min-confidence">Min confidence: {minConfidence.toFixed(1)}</label>
          <input
            id="min-confidence"
            type="range"
            min={0}
            max={1}
            step={0.1}
            value={minConfidence}
            onChange={(e) => setMinConfidence(Number(e.target.value))}
          />
        </div>
        <button
          className="btn btn-primary"
          onClick={process}
          disabled={processing || !image}
        >
          {processing ? "Processing…" : "Extract text"}
        </button>
        {status && <p className={`status ${status.kind}`}>{status.text}</p>}
      </section>

      {result && (
        <section className="card">
          <h2>Result</h2>
          <pre className="result-text">{result.text || "No text found"}</pre>
          <dl className="details">
            <dt>Model</dt>
            <dd>{result.model.label ?? result.model.id}</dd>
            <dt>Lines</dt>
            <dd>{result.line_count}</dd>
            <dt>Time</dt>
            <dd>{result.processing_time_ms} ms</dd>
            <dt>Image</dt>
            <dd>
              {result.image.width} × {result.image.height}
            </dd>
            {result.usage && (
              <>
                <dt>Tokens</dt>
                <dd>
                  {result.usage.input_tokens ?? 0} in / {result.usage.output_tokens ?? 0} out
                </dd>
                <dt>Cost</dt>
                <dd>
                  {result.usage.estimated_cost_usd === null
                    ? "unavailable"
                    : `$${result.usage.estimated_cost_usd.toFixed(6)}`}
                </dd>
              </>
            )}
          </dl>
        </section>
      )}
    </main>
  );
}
