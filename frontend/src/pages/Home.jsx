import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export default function Home() {
  const { user } = useAuth();

  return (
    <div className="p-16 grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
      <motion.div
        initial={{ x: -80, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.7 }}
      >
        <h2 className="text-5xl font-extrabold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-purple-600">
          Detect Fake News with Confidence
        </h2>
        <p className="text-gray-600 mb-8 text-lg leading-relaxed">
          TruthLens uses advanced transformer-based AI to classify news articles
          and visually explain why a prediction was made.
        </p>
        {user ? (
          <Link
            to="/analyze"
            className="bg-blue-600 hover:bg-blue-700 transition text-white px-8 py-4 rounded-2xl shadow-xl text-lg"
          >
            Start Analysis
          </Link>
        ) : (
          <div className="flex gap-4">
            <Link
              to="/signup"
              className="bg-blue-600 hover:bg-blue-700 transition text-white px-8 py-4 rounded-2xl shadow-xl text-lg"
            >
              Get Started
            </Link>
            <Link
              to="/login"
              className="bg-gray-100 hover:bg-gray-200 transition text-gray-700 px-8 py-4 rounded-2xl shadow text-lg"
            >
              Login
            </Link>
          </div>
        )}
      </motion.div>

      <motion.div
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.8 }}
        className="flex justify-center"
      >
        <img
          src="https://cdn-icons-png.flaticon.com/512/4712/4712109.png"
          alt="AI Avatar"
          className="w-80 drop-shadow-2xl"
        />
      </motion.div>
    </div>
  );
}