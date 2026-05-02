import { createContext, useContext, useState, useEffect } from "react";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user,  setUser]  = useState(null);
  const [token, setToken] = useState(() => localStorage.getItem("tl_token") || null);
  const [loading, setLoading] = useState(true);

  // On mount — verify token and load user
  useEffect(() => {
    if (!token) {
      setLoading(false);
      return;
    }
    fetch("http://localhost:5000/api/auth/me", {
      headers: { Authorization: `Bearer ${token}` }
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.id) setUser(data);
        else         logout();
      })
      .catch(() => logout())
      .finally(() => setLoading(false));
  }, []);

  function saveAuth(token, user) {
    localStorage.setItem("tl_token", token);
    setToken(token);
    setUser(user);
  }

  function logout() {
    localStorage.removeItem("tl_token");
    setToken(null);
    setUser(null);
  }

  // Attach token to every fetch automatically
  function authFetch(url, options = {}) {
    return fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...(options.headers || {})
      }
    });
  }

  return (
    <AuthContext.Provider value={{ user, token, loading, saveAuth, logout, authFetch }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}