import React, { useState } from "react";
import GestureControlPage from "./GestureControlPage";

function App() {
  const [start, setStart] = useState(false);

  if (!start) {
    return (
      <div style={styles.page}>
        <nav style={styles.navbar}>
          <div style={styles.logo}>Gesture Recognition</div>
          <div style={styles.navIcons}>
            <input style={styles.searchInput} type="text" placeholder="Search..." />
            <span style={styles.navIcon}>🔗</span>
            <span style={styles.navIcon}>👤</span>
          </div>
        </nav>

        <div style={styles.content}>
          <div style={styles.leftSection}>
            <h1 style={styles.heading}>
              Gesture <span style={{ color: "#0056D2" }}>Recognition</span>
            </h1>
            <p style={styles.subtext}>
              Control your media with hand gestures. Fast, intuitive, and futuristic. 
              Let's bring the future to your fingertips.
            </p>
            <button style={styles.getStartedButton} onClick={() => setStart(true)}>
              🚀 Get started now!
            </button>

            <div style={styles.blogLinks}>
              <span style={styles.blogLink}>Blog</span> | <span style={styles.blogLink}>About</span>
            </div>
          </div>

          <div style={styles.rightSection}>
            <img src="gesture-illustration.png" alt="Gesture Illustration" style={styles.image} />
          </div>
        </div>
      </div>
    );
  }

  return <GestureControlPage />;
}

const styles = {
  page: {
    fontFamily: "'Poppins', sans-serif",
    backgroundColor: "#f4f7fc",
    minHeight: "100vh",
    display: "flex",
    flexDirection: "column",
  },
  navbar: {
    display: "flex",
    justifyContent: "space-between",
    padding: "20px 40px",
    backgroundColor: "#ffffff",
    boxShadow: "0 2px 8px rgba(0, 0, 0, 0.05)",
    alignItems: "center",
  },
  logo: {
    fontSize: "24px",
    fontWeight: "600",
    color: "#333",
  },
  navIcons: {
    display: "flex",
    alignItems: "center",
    gap: "15px",
  },
  searchInput: {
    padding: "8px 12px",
    borderRadius: "8px",
    border: "1px solid #ddd",
    outline: "none",
  },
  navIcon: {
    fontSize: "20px",
    cursor: "pointer",
  },
  content: {
    display: "flex",
    flex: 1,
    padding: "60px 80px",
    alignItems: "center",
    justifyContent: "space-between",
  },
  leftSection: {
    maxWidth: "500px",
  },
  heading: {
    fontSize: "48px",
    fontWeight: "700",
    marginBottom: "20px",
    color: "#333",
  },
  subtext: {
    fontSize: "18px",
    color: "#555",
    marginBottom: "30px",
    lineHeight: "1.6",
  },
  getStartedButton: {
    padding: "15px 40px",
    backgroundColor: "#0056D2", // Deep Blue
    border: "none",
    color: "#fff",
    fontSize: "18px",
    borderRadius: "50px",
    cursor: "pointer",
    boxShadow: "0 4px 12px rgba(0, 86, 210, 0.4)",
    transition: "background-color 0.3s ease",
  },
  blogLinks: {
    marginTop: "30px",
    color: "#777",
    fontSize: "14px",
  },
  blogLink: {
    cursor: "pointer",
    textDecoration: "underline",
  },
  rightSection: {
    flex: 1,
    textAlign: "center",
  },
  image: {
    maxWidth: "100%",
    height: "auto",
  },
};

export default App;
