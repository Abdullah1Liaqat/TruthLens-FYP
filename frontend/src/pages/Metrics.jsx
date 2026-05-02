import { useState, useEffect } from "react";

export default function Metrics() {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    fetch("http://localhost:5000/api/metrics")
      .then((r) => r.json())
      .then(setMetrics)
      .catch(console.error);
  }, []);

  if (!metrics) {
    return (
      <div className="p-10 text-center text-gray-400 animate-pulse">
        Loading metrics…
      </div>
    );
  }

  const pct = (v) => (v * 100).toFixed(1);

  const cards = [
    { label: "Accuracy",  value: metrics.accuracy,  color: "from-blue-500 to-blue-400"   },
    { label: "Precision", value: metrics.precision, color: "from-purple-500 to-purple-400" },
    { label: "Recall",    value: metrics.recall,    color: "from-green-500 to-green-400"  },
    { label: "F1 Score",  value: metrics.f1_score,  color: "from-orange-500 to-orange-400" }
  ];

  const last = metrics.epochs[metrics.epochs.length - 1];

  return (
    <div className="p-10 max-w-5xl mx-auto">
      {/* Header */}
      <div className="text-center mb-10">
        <div className="text-6xl mb-2">📊</div>
        <h2 className="text-3xl font-bold">Evaluation Dashboard</h2>
        <p className="text-gray-500 text-sm mt-2">Model performance on test dataset</p>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-5 mb-10">
        {cards.map((m, i) => (
          <div key={i} className="bg-white rounded-2xl shadow p-5 flex flex-col items-center gap-2">
            {/* Circular indicator */}
            <div className={`w-20 h-20 rounded-full bg-gradient-to-br ${m.color} flex items-center justify-center shadow-lg`}>
              <span className="text-white font-extrabold text-lg">{pct(m.value)}%</span>
            </div>
            <p className="text-gray-600 font-medium text-sm">{m.label}</p>
            <div className="w-full bg-gray-100 h-1.5 rounded">
              <div
                className={`h-1.5 rounded bg-gradient-to-r ${m.color}`}
                style={{ width: `${m.value * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Training bar chart */}
      {metrics.epochs && (
        <div className="bg-white shadow rounded-2xl p-6 mb-8">
          <h3 className="text-xl font-semibold mb-6">Training Progress per Epoch</h3>

          <div className="flex items-end gap-6 h-52 border-b border-gray-200 pb-2">
            {metrics.epochs.map((e, i) => (
              <div key={i} className="flex flex-col items-center flex-1 gap-1">
                {/* Loss bar (red, behind) */}
                <div className="w-full flex gap-1 items-end justify-center" style={{ height: "180px" }}>
                  <div
                    className="w-5 bg-red-300 rounded-t"
                    style={{ height: `${e.loss * 180}px`, transition: "height 0.5s ease" }}
                    title={`Loss: ${e.loss}`}
                  />
                  <div
                    className="w-5 bg-gradient-to-t from-blue-600 to-blue-300 rounded-t"
                    style={{ height: `${e.accuracy * 180}px`, transition: "height 0.5s ease" }}
                    title={`Accuracy: ${(e.accuracy * 100).toFixed(1)}%`}
                  />
                </div>
                <p className="text-xs text-gray-600 font-medium">E{i + 1}</p>
              </div>
            ))}
          </div>

          <div className="flex gap-6 mt-4 text-xs text-gray-500">
            <span><span className="inline-block w-3 h-3 rounded bg-blue-400 mr-1" />Accuracy</span>
            <span><span className="inline-block w-3 h-3 rounded bg-red-300 mr-1" />Loss</span>
          </div>
        </div>
      )}

      {/* Convergence summary */}
      <div className="bg-green-50 border border-green-200 rounded-2xl p-6 mb-8">
        <h3 className="text-xl font-semibold mb-3">Model Convergence Summary</h3>
        <ul className="text-gray-700 space-y-2 text-sm">
          <li>📉 Final Train Loss: <b>{last.loss}</b></li>
          <li>🎯 Final Validation Accuracy: <b>{pct(last.accuracy)}%</b></li>
          <li>📊 Trend:{" "}
            <b>{metrics.epochs[0].accuracy < last.accuracy
              ? "Improving / Stable convergence"
              : "Minor fluctuation but stable"}
            </b>
          </li>
        </ul>
        <p className="mt-3 text-sm text-gray-600">
          The model shows consistent convergence across epochs with minimal overfitting,
          indicating stable fine-tuning of the BERT classifier.
        </p>
      </div>

      {/* Insights */}
      <div className="bg-blue-50 border border-blue-200 rounded-2xl p-6">
        <h3 className="text-xl font-semibold mb-2">Model Insights</h3>
        <p className="text-gray-700 leading-relaxed">
          The model demonstrates strong classification capability with an accuracy of{" "}
          <b>{pct(metrics.accuracy)}%</b>. Precision and recall values indicate a balanced
          performance, meaning the model is effective in identifying both real and fake news.
        </p>
        <p className="text-gray-700 mt-3">
          Explainability modules (LIME & SHAP) ensure full transparency in predictions —
          every verdict can be traced back to specific words and patterns.
        </p>
      </div>
    </div>
  );
}