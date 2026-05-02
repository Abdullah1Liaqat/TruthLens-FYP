import { motion } from "framer-motion";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export default function Navbar() {
  const auth = useAuth();
  const user = auth?.user || null;
  const logout = auth?.logout || (() => {});
  const navigate = useNavigate();

  function handleLogout() {
    logout();
    navigate("/login");
  }

  return (
    <motion.nav
      initial={{ y: -60, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6 }}
      className="bg-gradient-to-r from-gray-900 via-gray-800 to-gray-900 text-white px-8 py-5 flex justify-between items-center shadow-2xl rounded-b-2xl"
    >
      <Link to="/" className="text-xl font-bold tracking-wide text-white">
        TruthLens
      </Link>

      <div className="flex items-center gap-5">
        <Link to="/"        className="hover:text-blue-400 text-sm text-white">Home</Link>
        <Link to="/analyze" className="hover:text-blue-400 text-sm text-white">Analyze</Link>
        <Link to="/explain" className="hover:text-blue-400 text-sm text-white">Explainability</Link>
        <Link to="/metrics" className="hover:text-blue-400 text-sm text-white">Dashboard</Link>
        <Link to="/history" className="hover:text-green-400 text-sm text-white">History</Link>
        <Link to="/about"   className="hover:text-blue-400 text-sm text-white">About</Link>

        {user ? (
          <div className="flex items-center gap-3 ml-4 pl-4 border-l border-gray-600">
            <span className="text-sm text-gray-300">
              👤 <span className="font-semibold text-white">{user.username}</span>
            </span>
            <button
              onClick={handleLogout}
              className="text-xs bg-red-600 hover:bg-red-700 px-3 py-1.5 rounded-lg transition text-white"
            >
              Logout
            </button>
          </div>
        ) : (
          <div className="flex items-center gap-2 ml-4 pl-4 border-l border-gray-600">
            <Link to="/login"  className="text-xs bg-gray-700 hover:bg-gray-600 px-3 py-1.5 rounded-lg transition text-white">Login</Link>
            <Link to="/signup" className="text-xs bg-blue-600 hover:bg-blue-700 px-3 py-1.5 rounded-lg transition text-white">Sign Up</Link>
          </div>
        )}
      </div>
    </motion.nav>
  );
}