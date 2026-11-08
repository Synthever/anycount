import React, { useEffect, useState } from "react";
import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom";
import { api } from "./api/client";
import { Sidebar } from "./components/Sidebar";
import { Topbar } from "./components/Topbar";
import { PasscodeModal } from "./components/PasscodeModal";
import { PlaygroundPage } from "./pages/PlaygroundPage";
import { BenchmarkPage } from "./pages/BenchmarkPage";
import { ModelPacksPage } from "./pages/ModelPacksPage";
import { DocsGlossaryPage } from "./pages/DocsGlossaryPage";

function getPageTitle(pathname: string): string {
  if (pathname.startsWith("/benchmark")) return "Benchmark Gallery";
  if (pathname.startsWith("/packs")) return "Model Packs (.onnx)";
  if (pathname.startsWith("/docs")) return "Method Docs & Glossary";
  return "Playground";
}

const StudioLayout: React.FC = () => {
  const location = useLocation();
  const [dark, setDark] = useState<boolean>(() => {
    if (typeof document !== "undefined") {
      return document.documentElement.classList.contains("dark");
    }
    return true;
  });
  const [collapsed, setCollapsed] = useState<boolean>(false);
  const [unlocked, setUnlocked] = useState<boolean>(false);
  const [unlockModalOpen, setUnlockModalOpen] = useState<boolean>(false);
  const [packCount, setPackCount] = useState<number>(1);

  const refreshPacksCount = () => {
    api
      .getPacks()
      .then((res) => {
        const ready = res.packs.filter((p) => p.status === "ready").length;
        setPackCount(ready || res.packs.length || 1);
      })
      .catch(() => {});
  };

  useEffect(() => {
    if (!document.documentElement.classList.contains("light-cream")) {
      document.documentElement.classList.add("dark");
      setDark(true);
    }
    api
      .getAuthStatus()
      .then((res) => setUnlocked(res.unlocked))
      .catch(() => {});
    refreshPacksCount();
  }, []);

  const handleToggleTheme = () => {
    const nextDark = !dark;
    setDark(nextDark);
    if (nextDark) {
      document.documentElement.classList.add("dark");
      document.documentElement.classList.remove("light-cream");
    } else {
      document.documentElement.classList.remove("dark");
      document.documentElement.classList.add("light-cream");
    }
  };

  const handleLogout = async () => {
    try {
      await api.logout();
      setUnlocked(false);
    } catch {
      // ignore
    }
  };

  return (
    <div
      className="min-h-screen flex"
      style={{
        backgroundColor: "var(--canvas)",
        color: "var(--ink)",
      }}
    >
      <Sidebar collapsed={collapsed} packCount={packCount} />

      <div className="flex-1 flex flex-col min-w-0">
        <Topbar
          pageTitle={getPageTitle(location.pathname)}
          dark={dark}
          onToggleTheme={handleToggleTheme}
          unlocked={unlocked}
          onOpenUnlock={() => setUnlockModalOpen(true)}
          onToggleSidebar={() => setCollapsed((c) => !c)}
          onLogout={handleLogout}
        />

        <main className="flex-1">
          <Routes>
            <Route
              path="/"
              element={
                <PlaygroundPage
                  unlocked={unlocked}
                  onRequireUnlock={() => setUnlockModalOpen(true)}
                  onPacksChanged={refreshPacksCount}
                />
              }
            />
            <Route path="/benchmark" element={<BenchmarkPage />} />
            <Route path="/benchmarks" element={<BenchmarkPage />} />
            <Route
              path="/packs"
              element={
                <ModelPacksPage
                  unlocked={unlocked}
                  onRequireUnlock={() => setUnlockModalOpen(true)}
                  onPacksChanged={(c) => setPackCount(c)}
                />
              }
            />
            <Route path="/docs" element={<DocsGlossaryPage />} />
          </Routes>
        </main>
      </div>

      <PasscodeModal
        open={unlockModalOpen}
        onClose={() => setUnlockModalOpen(false)}
        onUnlocked={() => setUnlocked(true)}
      />
    </div>
  );
};

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <StudioLayout />
    </BrowserRouter>
  );
};

export default App;
