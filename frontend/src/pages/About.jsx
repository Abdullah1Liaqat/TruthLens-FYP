import { useState, useEffect } from "react";

export default function About() {
  const [info, setInfo] = useState(null);

  useEffect(() => {
    fetch("http://localhost:5000/api/about")
      .then((r) => r.json())
      .then(setInfo)
      .catch(console.error);
  }, []);

  if (!info) {
    return (
      <div className="p-10 text-center text-gray-400 animate-pulse">Loading…</div>
    );
  }

  return (
    <div className="p-10 max-w-5xl mx-auto space-y-10">
      <h2 className="text-3xl font-bold">About {info.project}</h2>

      {/* Description */}
      <div className="bg-gray-50 border border-gray-200 p-6 rounded-xl shadow-sm">
        <h3 className="text-xl font-semibold mb-2">Project Overview</h3>
        <p className="text-gray-600">{info.description}</p>
      </div>

      {/* Research */}
      <div className="bg-white border border-gray-200 p-6 rounded-xl shadow-sm">
        <h3 className="text-xl font-semibold mb-3">Research Perspective</h3>
        <p><b>Problem:</b> {info.research.problem}</p>
        <p className="mt-2"><b>Research Gap:</b> {info.research.gap}</p>
        <p className="mt-2"><b>Objective:</b> {info.research.objective}</p>
        <ul className="mt-3 list-disc pl-6 text-gray-600 space-y-1">
          {info.research.approach.map((a, i) => <li key={i}>{a}</li>)}
        </ul>
      </div>

      {/* Datasets */}
      <div className="bg-gray-50 border border-gray-200 p-6 rounded-xl shadow-sm">
        <h3 className="text-xl font-semibold mb-3">Datasets Used</h3>
        <ul className="list-disc pl-6 text-gray-600 space-y-1">
          {info.datasets.map((d, i) => <li key={i}>{d}</li>)}
        </ul>
      </div>

      {/* Tech Stack */}
      <div className="bg-blue-50 border border-blue-100 p-6 rounded-xl shadow-sm">
        <h3 className="text-xl font-semibold mb-4">Tech Stack</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[
            { label: "BERT",        icon: "🤖", desc: "Transformer model"   },
            { label: "Flask",       icon: "🐍", desc: "Backend API"         },
            { label: "React",       icon: "⚛️",  desc: "Frontend UI"         },
            { label: "LIME",        icon: "🍋", desc: "Local explainability" },
            { label: "SHAP",        icon: "📐", desc: "Global explainability"},
            { label: "SQLite",      icon: "🗃️",  desc: "Database"            },
            { label: "NewsAPI",     icon: "📰", desc: "News context"         },
            { label: "Fact Check",  icon: "✅", desc: "Google fact-check"    }
          ].map((t, i) => (
            <div key={i} className="bg-white border border-blue-100 rounded-lg p-3 text-center shadow-sm">
              <div className="text-2xl mb-1">{t.icon}</div>
              <p className="font-semibold text-sm">{t.label}</p>
              <p className="text-xs text-gray-500">{t.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Contributors */}
      <div className="bg-green-50 border border-green-100 p-6 rounded-xl shadow-sm">
        <h3 className="text-xl font-semibold mb-4">Contributors</h3>
        <div className="grid md:grid-cols-3 gap-4">
          {info.contributors.map((c, i) => (
            <div key={i} className="bg-white border border-gray-200 p-4 rounded-lg shadow-sm hover:shadow-md transition">
              <div className="text-3xl mb-2">
                {i === 0 ? "👨‍💻" : i === 1 ? "🧑‍💻" : "🎓"}
              </div>
              <p className="font-bold">{c.name}</p>
              <p className="text-blue-600 text-sm">{c.role}</p>
              <p className="text-xs text-gray-500 mt-1">{c.work}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}