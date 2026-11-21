import { createContext, useCallback, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";
import { clearSession, fetchMe, getStoredUser, getToken, setSession } from "./api";
import type { User } from "./api";
import { login as apiLogin } from "./api";

interface AuthState {
  user: User | null;
  ready: boolean;
  signIn: (username: string, password: string) => Promise<void>;
  signOut: () => void;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(getStoredUser());
  const [ready, setReady] = useState(false);

  useEffect(() => {
    // Validate a stored session against the server once on load.
    if (!getToken()) {
      setUser(null);
      setReady(true);
      return;
    }
    fetchMe()
      .then((me) => {
        if (me.kind === "user" && me.user) {
          setUser(me.user);
          setSession(getToken() as string, me.user);
        } else {
          clearSession();
          setUser(null);
        }
      })
      .catch(() => {
        clearSession();
        setUser(null);
      })
      .finally(() => setReady(true));
  }, []);

  const signIn = useCallback(async (username: string, password: string) => {
    const { user } = await apiLogin(username, password);
    setUser(user);
  }, []);

  const signOut = useCallback(() => {
    clearSession();
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider value={{ user, ready, signIn, signOut }}>{children}</AuthContext.Provider>
  );
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside AuthProvider");
  return ctx;
}
