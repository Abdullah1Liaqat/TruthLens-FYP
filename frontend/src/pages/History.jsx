import { useState, useEffect } from "react";
import { useAuth } from "../context/AuthContext";

export default function History() {
  const { authFetch, user } = useAuth();
  const [data,    setData]    = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    authFetch("http://localhost:5000/api/history")
      .then((r) => r.json())
      .then((json) => {
        if (Array.isArray(json))      setData(json);
        else if (Array.isArray(json.data)) setData(json.data);
        else                          setData([]);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  async function handleDelete(id) {
    try {
      await authFetch(`http://localhost:5000/api/history/${id}`, { method: "DELETE" });
      setData((prev) => prev.filter((item) => item.id !== id));
    } catch (err) {
      console.error(err);
    }
  }

  const methodLabel = {
    fact_check: "Fact Check",
    newsapi:    "NewsAPI",
    model:      "AI Model"
  };

  const methodBadge = {
    fact_check: "bg-purple-100 text-purple-700",
    newsapi:    "bg-blue-100   text-blue-700",
    model:      "bg-gray-100   text-gray-600"
  };

  return (
    <div className="p-10 max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-3xl font-bold">History</h2>
          {user && (
            <p className="text-gray-500 text-sm mt-1">
              Showing predictions for <b>{user.username}</b>
            </p>
          )}
        </div>
        {data.length > 0 && (
          <span className="text-sm text-gray-400">{data.length} record{data.length !== 1 ? "s" : ""}</span>
        )}
      </div>

      {loading && (
        <div className="space-y-3 animate-pulse">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="h-12 bg-gray-100 rounded-lg" />
          ))}
        </div>
      )}

      {!loading && data.length === 0 && (
        <div className="text-center py-16 text-gray-400">
          <div className="text-5xl mb-3">📭</div>
          <p>No history yet. Start by analyzing an article.</p>
        </div>
      )}

      {!loading && data.length > 0 && (
        <div className="overflow-x-auto rounded-xl border border-gray-200 shadow-sm">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-100 text-gray-600 text-left">
                <th className="px-4 py-3 font-semibold">#</th>
                <th className="px-4 py-3 font-semibold">Text</th>
                <th className="px-4 py-3 font-semibold text-center">Label</th>
                <th className="px-4 py-3 font-semibold text-center">Confidence</th>
                <th className="px-4 py-3 font-semibold text-center">Verified By</th>
                <th className="px-4 py-3 font-semibold text-center">Action</th>
              </tr>
            </thead>
            <tbody>
              {data.map((item, idx) => (
                <tr key={item.id} className="border-t border-gray-100 hover:bg-gray-50 transition">
                  <td className="px-4 py-3 text-gray-400">{idx + 1}</td>

                  <td className="px-4 py-3 max-w-xs">
                    <p className="truncate text-gray-700" title={item.text}>{item.text}</p>
                  </td>

                  <td className="px-4 py-3 text-center">
                    <span className={`inline-block font-bold text-sm px-2 py-0.5 rounded-full ${
                      item.label === "FAKE"
                        ? "bg-red-100 text-red-700"
                        : item.label === "REAL"
                        ? "bg-green-100 text-green-700"
                        : "bg-yellow-100 text-yellow-700"
                    }`}>
                      {item.label === "FAKE" ? "❌ FAKE" : item.label === "REAL" ? "✅ REAL" : "⚠️ UNCERTAIN"}
                    </span>
                  </td>

                  <td className="px-4 py-3 text-center text-gray-500">
                    {item.confidence != null
                      ? `${(item.confidence * 100).toFixed(1)}%`
                      : "—"}
                  </td>

                  <td className="px-4 py-3 text-center">
                    <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${
                      methodBadge[item.verification_method] || methodBadge.model
                    }`}>
                      {methodLabel[item.verification_method] || "AI Model"}
                    </span>
                  </td>

                  <td className="px-4 py-3 text-center">
                    <button
                      onClick={() => handleDelete(item.id)}
                      className="text-xs bg-red-500 hover:bg-red-600 text-white px-3 py-1 rounded-lg transition"
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}