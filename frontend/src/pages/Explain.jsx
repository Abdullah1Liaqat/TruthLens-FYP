import { useState } from "react";
import { useAuth } from "../context/AuthContext";

export default function Explain() {
  const { authFetch } = useAuth();
  const [mode,     setMode]     = useState("lime");
  const [limeData, setLimeData] = useState([]);
  const [shapData, setShapData] = useState([]);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState("");

  const text = localStorage.getItem("truthlens_text") || "";

  if (!text) {
    return (
      <p className="p-10 text-center text-gray-500">
        No analyzed text found. Please analyze news first.
      </p>
    );
  }

  async function generateLime() {
    setLoading(true);
    setError("");
    try {
      const res  = await authFetch("http://localhost:5000/api/lime", {
        method: "POST",
        body:   JSON.stringify({ text })
      });
      const json = await res.json();
      setLimeData(json.data || []);
      setMode("lime");
    } catch (err) {
      setError("Failed to generate LIME explanation.");
    } finally {
      setLoading(false);
    }
  }

  async function loadShap() {
    setLoading(true);
    setError("");
    try {
      const res  = await authFetch("http://localhost:5000/api/shap", {
        method: "POST",
        body:   JSON.stringify({ text })
      });
      const json = await res.json();
      setShapData(json.data || []);
      setMode("shap");
    } catch (err) {
      setError("Failed to generate SHAP explanation.");
    } finally {
      setLoading(false);
    }
  }

  const activeData = mode === "lime" ? limeData : shapData;

  const maxAbs = activeData.length
    ? Math.max(...activeData.map((d) => Math.abs(d.value)), 0.001)
    : 1;

  return (
    <div className="p-10 max-w-4xl mx-auto">
      <div className="text-center mb-8">
        <div className="text-7xl mb-3">🔍</div>
        <h2 className="text-3xl font-bold">
          Explainability — {mode === "lime" ? "LIME" : "SHAP"}
        </h2>
        <p className="text-gray-500 text-sm mt-2">
          Understand which words influenced the model's decision
        </p>
      </div>

      {/* Analyzed text preview */}
      <div className="mb-6 bg-gray-50 border border-gray-200 rounded-xl px-4 py-3 text-sm text-gray-600">
        <span className="font-semibold text-gray-800">Analyzed text: </span>
        <span className="italic">{text.length > 200 ? text.slice(0, 200) + "…" : text}</span>
      </div>

      {/* Buttons */}
      <div className="flex gap-4 mb-6 justify-center">
        <button
          onClick={generateLime}
          disabled={loading}
          className="bg-purple-600 hover:bg-purple-700 disabled:opacity-50 text-white px-6 py-2 rounded-xl shadow transition"
        >
          {loading && mode === "lime" ? "Generating…" : "LIME"}
        </button>
        <button
          onClick={loadShap}
          disabled={loading}
          className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white px-6 py-2 rounded-xl shadow transition"
        >
          {loading && mode === "shap" ? "Generating…" : "SHAP"}
        </button>
      </div>

      {error && (
        <p className="text-center text-red-500 text-sm mb-4">{error}</p>
      )}

      {loading && (
        <div className="space-y-2 animate-pulse">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="h-8 bg-gray-100 rounded-lg" />
          ))}
        </div>
      )}

      {!loading && activeData.length > 0 && (
        <div className="bg-white border border-gray-200 p-6 rounded-xl shadow space-y-2">
          {/* Legend */}
          <div className="flex gap-6 text-xs text-gray-500 mb-4">
            <span>
              <span className="inline-block w-3 h-3 rounded bg-green-400 mr-1" />
              Pushes toward REAL
            </span>
            <span>
              <span className="inline-block w-3 h-3 rounded bg-red-400 mr-1" />
              Pushes toward FAKE
            </span>
          </div>

          {activeData.map((item, idx) => {
            const value    = item.value ?? 0;
            const isPos    = value > 0;
            const barWidth = `${(Math.abs(value) / maxAbs) * 100}%`;

            return (
              <div key={idx} className="flex items-center gap-3">
                {/* Token */}
                <span className="w-28 text-sm font-mono text-gray-700 truncate text-right shrink-0">
                  {item.token}
                </span>

                {/* Bar */}
                <div className="flex-1 flex items-center gap-2">
                  <div className="flex-1 bg-gray-100 rounded h-5 relative overflow-hidden">
                    <div
                      className={`h-5 rounded transition-all ${isPos ? "bg-green-400" : "bg-red-400"}`}
                      style={{ width: barWidth }}
                    />
                  </div>
                  <span className={`text-xs font-semibold w-16 text-right ${isPos ? "text-green-700" : "text-red-700"}`}>
                    {isPos ? "+" : ""}{value.toFixed(4)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {!loading && activeData.length === 0 && (
        <div className="text-center text-gray-400 py-10">
          Click <b>LIME</b> or <b>SHAP</b> to generate an explanation.
        </div>
      )}
    </div>
  );
}