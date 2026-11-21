import { useState } from "react";
import type { FormEvent } from "react";
import { Link } from "react-router-dom";
import { ApiError, register } from "../api";

export default function RegisterPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const detail = await register(username, password);
      setDone(detail);
    } catch (err) {
      if (err instanceof ApiError) setError(err.message);
      else setError("Registration failed. Please try again.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="page auth-page">
      <section className="card">
        <h1>Create account</h1>
        {done ? (
          <>
            <p className="status success">{done}</p>
            <p className="muted">
              Once approved, <Link to="/login">sign in</Link>.
            </p>
          </>
        ) : (
          <>
            <p className="muted">New accounts need administrator approval.</p>
            <form onSubmit={onSubmit}>
              <label htmlFor="username">Username</label>
              <input
                id="username"
                autoComplete="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
                minLength={3}
                maxLength={32}
              />
              <label htmlFor="password">Password</label>
              <input
                id="password"
                type="password"
                autoComplete="new-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                minLength={8}
              />
              {error && <p className="status error">{error}</p>}
              <button type="submit" className="btn btn-primary" disabled={busy}>
                {busy ? "Creating…" : "Register"}
              </button>
            </form>
            <p className="muted">
              Already have an account? <Link to="/login">Sign in</Link>
            </p>
          </>
        )}
      </section>
    </main>
  );
}
