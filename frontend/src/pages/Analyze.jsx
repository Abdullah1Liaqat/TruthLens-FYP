import { useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { useAuth } from "../context/AuthContext";

// ─── Sub-components ───────────────────────────────────────────────────────────

function MethodBadge({ method }) {
  const config = {
    fact_check: { text: "Fact-Check Database",  bg: "bg-purple-100 text-purple-700 border-purple-300" },
    newsapi:    { text: "News Source Database", bg: "bg-blue-100   text-blue-700   border-blue-300"   },
    model:      { text: "AI Model (BERT)",      bg: "bg-gray-100   text-gray-700   border-gray-300"   }
  };
  const c = config[method] || config.model;
  return (
    <span className={`inline-block text-xs font-semibold px-3 py-1 rounded-full border ${c.bg}`}>
      🔎 Verdict by: {c.text}
    </span>
  );
}

function NewsSourceCards({ sources, isVerdict = false }) {
  if (!sources || sources.length === 0) return null;
  return (
    <div>
      <p className="text-sm font-semibold text-gray-700 mb-2">
        {isVerdict ? "📰 Verified by Matching News Articles:" : "📰 Related Coverage (context):"}
      </p>
      <div className="space-y-2">
        {sources.map((s, i) => (
          <a key={i} href={s.url} target="_blank" rel="noreferrer"
            className={`block border rounded-lg px-4 py-2 transition ${
              isVerdict
                ? "border-blue-200 bg-blue-50 hover:bg-blue-100"
                : "border-gray-200 bg-gray-50 hover:bg-gray-100"
            }`}>
            <span className={`font-semibold text-sm ${isVerdict ? "text-blue-700" : "text-gray-700"}`}>
              {s.name}
            </span>
            {s.title && <span className="block text-xs text-gray-500 mt-0.5 truncate">{s.title}</span>}
          </a>
        ))}
      </div>
    </div>
  );
}

function FactCheckCards({ factChecks }) {
  if (!factChecks || factChecks.length === 0) return null;
  const style = {
    FAKE:      "bg-red-50    border-red-300   text-red-700",
    REAL:      "bg-green-50  border-green-300 text-green-700",
    UNCERTAIN: "bg-yellow-50 border-yellow-300 text-yellow-700"
  };
  return (
    <div>
      <p className="text-sm font-semibold text-gray-700 mb-2">🏷️ Fact-Check Results:</p>
      <div className="space-y-2">
        {factChecks.map((fc, i) => (
          <a key={i} href={fc.url || "#"} target="_blank" rel="noreferrer"
            className={`block border rounded-lg px-4 py-2 hover:opacity-90 transition ${style[fc.verdict] || style.UNCERTAIN}`}>
            <div className="flex justify-between items-center">
              <span className="font-semibold text-sm">{fc.publisher}</span>
              <span className="text-xs font-bold uppercase tracking-wide">{fc.rating}</span>
            </div>
            {fc.claim_text && <p className="text-xs mt-1 opacity-80 truncate">"{fc.claim_text}"</p>}
          </a>
        ))}
      </div>
    </div>
  );
}

function RuleExplanation({ rules, score }) {
  if ((!rules || rules.length === 0) && score == null) return null;
  return (
    <div className="bg-gray-50 border border-gray-200 rounded-lg px-4 py-3 text-xs text-gray-600 space-y-1">
      <p className="font-semibold">Rule Engine</p>
      {score != null && (
        <p>
          Score:{" "}
          <span className={`font-bold ${score > 0 ? "text-green-600" : score < 0 ? "text-red-600" : "text-gray-500"}`}>
            {score > 0 ? `+${score}` : score}
          </span>
          <span className="text-gray-400 ml-1">
            ({score > 0 ? "credibility signals" : score < 0 ? "misinformation signals" : "neutral"})
          </span>
        </p>
      )}
      {rules && rules.map((r, i) => <p key={i}>• {r}</p>)}
    </div>
  );
}

function ConfidenceBar({ label, confidence }) {
  if (confidence == null) return null;
  return (
    <div>
      <p className="text-sm text-gray-500">
        Confidence:{" "}
        <span className="font-semibold">{(confidence * 100).toFixed(1)}%</span>
      </p>
      <div className="w-full bg-gray-200 h-2 rounded mt-1">
        <div
          className={`h-2 rounded transition-all ${label === "FAKE" ? "bg-red-500" : "bg-green-500"}`}
          style={{ width: `${confidence * 100}%` }}
        />
      </div>
    </div>
  );
}

// ─── Main page ────────────────────────────────────────────────────────────────
export default function Analyze() {
  const { authFetch } = useAuth();
  const [text,    setText]    = useState("");
  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);

  async function handleAnalyze() {
    if (!text.trim()) return;
    setLoading(true);
    setResult(null);
    try {
      const res  = await authFetch("http://localhost:5000/api/predict", {
        method: "POST",
        body:   JSON.stringify({ text })
      });
      const json = await res.json();
      setResult(json.data);
      localStorage.setItem("truthlens_text", text);
    } catch (err) {
      console.error(err);
    }
    setLoading(false);
  }

  const labelColor = (lbl) =>
    lbl === "FAKE" ? "text-red-600" : lbl === "REAL" ? "text-green-600" : "text-yellow-600";
  const labelIcon  = (lbl) =>
    lbl === "FAKE" ? "❌" : lbl === "REAL" ? "✅" : "⚠️";

  return (
    <div className="p-10 max-w-3xl mx-auto">
      <h2 className="text-3xl font-bold mb-4">Analyze News Article</h2>

      <textarea
        className="w-full border rounded-xl p-4 h-40 focus:outline-none focus:ring-2 focus:ring-blue-300"
        placeholder="Paste news text or headline here..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      <button
        onClick={handleAnalyze}
        disabled={loading}
        className="mt-4 bg-green-600 hover:bg-green-700 disabled:opacity-50 text-white px-6 py-2 rounded-xl transition"
      >
        {loading ? "Analyzing..." : "Analyze"}
      </button>

      {/* Loading skeleton */}
      {loading && (
        <div className="mt-6 p-4 border rounded-xl animate-pulse bg-gray-50">
          <div className="h-4 bg-gray-200 rounded w-1/3 mb-3" />
          <div className="h-6 bg-gray-200 rounded w-1/2 mb-3" />
          <div className="h-3 bg-gray-200 rounded w-2/3" />
        </div>
      )}

      {/* Result card */}
      {result && !loading && (() => {
        const apiLabel  = result.label;
        const bertLabel = result.bert_label;
        const method    = result.verification_method;
        const hasApi    = method === "fact_check" || method === "newsapi";
        const conflict  = hasApi && bertLabel && bertLabel !== apiLabel && apiLabel !== "UNCERTAIN";
        const agree     = hasApi && bertLabel && bertLabel === apiLabel;

        const methodLabel = {
          fact_check: "Fact-Check Verdict",
          newsapi:    "News Source Verdict",
          model:      "AI Model (BERT) Prediction"
        };

        return (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="mt-6 p-5 border rounded-xl shadow-sm bg-white space-y-4"
          >
            {/* CASE 1 — CONFLICT */}
            {conflict && (
              <>
                <div className="flex items-center gap-3">
                  <span className="text-4xl">{labelIcon(apiLabel)}</span>
                  <div>
                    <p className="text-xs text-gray-400 font-medium uppercase tracking-wide">
                      {methodLabel[method] || "External Verdict"}
                    </p>
                    <p className={`font-extrabold text-2xl ${labelColor(apiLabel)}`}>{apiLabel}</p>
                  </div>
                </div>
                <FactCheckCards factChecks={result.fact_checks} />
                <NewsSourceCards sources={result.sources} isVerdict={method === "newsapi"} />
                <hr className="border-gray-200" />
                <div className="flex items-center gap-3">
                  <span className="text-3xl">{labelIcon(bertLabel)}</span>
                  <div>
                    <p className="text-xs text-gray-400 font-medium uppercase tracking-wide">AI Model (BERT) Prediction</p>
                    <p className={`font-bold text-xl ${labelColor(bertLabel)}`}>{bertLabel}</p>
                  </div>
                </div>
                <ConfidenceBar label={bertLabel} confidence={result.bert_confidence} />
                <RuleExplanation rules={result.rule_explanation} score={result.rule_score} />
              </>
            )}

            {/* CASE 2 — AGREE */}
            {agree && (
              <>
                <div className="flex items-center gap-3">
                  <span className="text-4xl">{labelIcon(bertLabel)}</span>
                  <div>
                    <p className="text-xs text-gray-400 font-medium uppercase tracking-wide">AI Model (BERT) Prediction</p>
                    <p className={`font-extrabold text-2xl ${labelColor(bertLabel)}`}>{bertLabel}</p>
                  </div>
                </div>
                <ConfidenceBar label={bertLabel} confidence={result.bert_confidence} />
                <RuleExplanation rules={result.rule_explanation} score={result.rule_score} />
                <div className="flex items-start gap-2 bg-green-50 border border-green-200 rounded-lg px-4 py-3 text-sm text-green-800">
                  <span>✅</span>
                  <p>
                    <b>{method === "fact_check" ? "Fact-Check database" : "News sources"}</b>{" "}
                    confirm the same verdict: <b>{apiLabel}</b>.
                  </p>
                </div>
                <FactCheckCards factChecks={result.fact_checks} />
                <NewsSourceCards sources={result.sources} isVerdict={method === "newsapi"} />
              </>
            )}

            {/* CASE 3 — MODEL ONLY */}
            {!conflict && !agree && (
              <>
                <div className="flex items-center gap-3">
                  <span className="text-4xl">{labelIcon(apiLabel)}</span>
                  <div>
                    <p className="text-xs text-gray-400 font-medium uppercase tracking-wide">AI Model (BERT) Prediction</p>
                    <p className={`font-extrabold text-2xl ${labelColor(apiLabel)}`}>{apiLabel}</p>
                  </div>
                </div>
                <ConfidenceBar label={apiLabel} confidence={result.bert_confidence} />
                <div className="text-xs text-gray-400 bg-gray-50 border border-gray-200 rounded-lg px-3 py-2">
                  No matching fact-checks or news articles found. Verdict is based on AI model only.
                </div>
                <RuleExplanation rules={result.rule_explanation} score={result.rule_score} />
              </>
            )}

            <div className="pt-2 border-t border-gray-100">
              <Link to="/explain" className="text-sm text-blue-600 hover:underline">
                🔍 View LIME / SHAP Explanation →
              </Link>
            </div>
          </motion.div>
        );
      })()}
    </div>
  );
}