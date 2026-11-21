import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { deleteUser, fetchUsers, setUserStatus } from "../api";
import type { User } from "../api";
import { useAuth } from "../auth";

export default function AdminPage() {
  const { user, signOut } = useAuth();
  const [users, setUsers] = useState<User[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const reload = useCallback(() => {
    fetchUsers()
      .then(setUsers)
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load users"));
  }, []);

  useEffect(reload, [reload]);

  async function act(action: () => Promise<unknown>) {
    setBusy(true);
    setError(null);
    try {
      await action();
      reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Action failed");
    } finally {
      setBusy(false);
    }
  }

  const pending = users.filter((u) => u.status === "pending");

  return (
    <main className="page">
      <header className="topbar">
        <h1>User management</h1>
        <div className="topbar-actions">
          <Link to="/">← OCR</Link>
          <button className="btn" onClick={signOut}>Sign out</button>
        </div>
      </header>

      {error && <p className="status error">{error}</p>}

      <section className="card">
        <h2>Pending approval ({pending.length})</h2>
        {pending.length === 0 && <p className="muted">Nobody is waiting.</p>}
        {pending.map((entry) => (
          <div className="user-row" key={entry.id}>
            <span>{entry.username}</span>
            <span className="muted">registered {entry.created_at.slice(0, 19)}Z</span>
            <button
              className="btn btn-primary"
              disabled={busy}
              onClick={() => act(() => setUserStatus(entry.id, "approve"))}
            >
              Approve
            </button>
            <button
              className="btn btn-danger"
              disabled={busy}
              onClick={() => act(() => deleteUser(entry.id))}
            >
              Reject
            </button>
          </div>
        ))}
      </section>

      <section className="card">
        <h2>All users</h2>
        <table className="user-table">
          <thead>
            <tr>
              <th>Username</th>
              <th>Role</th>
              <th>Status</th>
              <th>Last sign-in</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {users.map((entry) => (
              <tr key={entry.id}>
                <td>{entry.username}</td>
                <td>{entry.role}</td>
                <td>
                  <span className={`badge ${entry.status}`}>{entry.status}</span>
                </td>
                <td>{entry.last_login_at ? `${entry.last_login_at.slice(0, 19)}Z` : "—"}</td>
                <td className="user-actions">
                  {entry.id !== user?.id && entry.status !== "approved" && (
                    <button
                      className="btn"
                      disabled={busy}
                      onClick={() => act(() => setUserStatus(entry.id, "approve"))}
                    >
                      Enable
                    </button>
                  )}
                  {entry.id !== user?.id && entry.status === "approved" && (
                    <button
                      className="btn"
                      disabled={busy}
                      onClick={() => act(() => setUserStatus(entry.id, "disable"))}
                    >
                      Disable
                    </button>
                  )}
                  {entry.id !== user?.id && (
                    <button
                      className="btn btn-danger"
                      disabled={busy}
                      onClick={() => {
                        if (window.confirm(`Delete ${entry.username}?`)) {
                          void act(() => deleteUser(entry.id));
                        }
                      }}
                    >
                      Delete
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </main>
  );
}
