import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider, useAuth } from "./context/AuthContext";

import Navbar   from "./components/Navbar";
import Home     from "./pages/Home";
import Analyze  from "./pages/Analyze";
import Explain  from "./pages/Explain";
import Metrics  from "./pages/Metrics";
import History  from "./pages/History";
import About    from "./pages/About";
import Login    from "./pages/Login";
import Signup   from "./pages/Signup";
console.log("COMPONENTS:", { 
  Home, Analyze, Explain, Metrics, 
  History, About, Login, Signup, Navbar 
});
console.log("Home:", typeof Home);
console.log("Analyze:", typeof Analyze);
console.log("Explain:", typeof Explain);
console.log("Metrics:", typeof Metrics);
console.log("History:", typeof History);
console.log("About:", typeof About);
console.log("Login:", typeof Login);
console.log("Signup:", typeof Signup);
console.log("Navbar:", typeof Navbar);
// Redirect to login if not authenticated
function ProtectedRoute({ children }) {
  const { user, loading } = useAuth();
  if (loading) return <div className="p-10 text-center text-gray-400">Loading…</div>;
  return user ? children : <Navigate to="/login" replace />;
}

function AppRoutes() {
  return (
    <>
      <Navbar />
      <Routes>
        {/* Public */}
        <Route path="/"       element={<Home />}   />
        <Route path="/login"  element={<Login />}  />
        <Route path="/signup" element={<Signup />} />
        <Route path="/about"  element={<About />}  />
        <Route path="/metrics" element={<Metrics />} />

        {/* Protected */}
        <Route path="/analyze" element={<ProtectedRoute><Analyze /></ProtectedRoute>} />
        <Route path="/explain" element={<ProtectedRoute><Explain /></ProtectedRoute>} />
        <Route path="/history" element={<ProtectedRoute><History /></ProtectedRoute>} />

        {/* Fallback */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <Router>
        <AppRoutes />
      </Router>
    </AuthProvider>
  );
}