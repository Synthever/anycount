import React, { useEffect, useState } from "react";
import { BrowserRouter, Routes, Route, Navigate, useLocation } from "react-router-dom";
import { api } from "./api/client";
import { Sidebar } from "./components/Sidebar";
import { Topbar } from "./components/Topbar";
import { PasscodeModal } from "./components/PasscodeModal";
import { PlaygroundPage } from "./pages/PlaygroundPage";
import { BenchmarkPage } from "./pages/BenchmarkPage";

function getPageTitle(pathname: string): string {
  if (pathname.startsWith("/benchmark")) return "Benchmark Gallery";
  return "Playground";
}

const THEME_KEY = "anycount_theme";

export function applyTheme(dark: boolean) {
  const root = document.documentElement;
  root.classList.toggle("dark", dark);
  root.classList.toggle("light-cream", !dark);
}

export function loadThemePreference(): boolean {
  try {
    const saved = window.localStorage.getItem(THEME_KEY);
    if (saved === "cream") return false;
    if (saved === "dark") return true;
  } catch {
    // private mode / storage disabled: fall through to the current class
  }
  return !document.documentElement.classList.contains("light-cream");
}

export function persistThemePreference(dark: boolean) {
  try {
    window.localStorage.setItem(THEME_KEY, dark ? "dark" : "cream");
  } catch {
    // ignore storage failures
  }
}

const StudioLayout: React.FC = () => {
  const location = useLocation();
  const [dark, setDark] = useState<boolean>(() => loadThemePreference());
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
    applyTheme(dark);
    persistThemePreference(dark);
  }, [dark]);

  useEffect(() => {
    api
      .getAuthStatus()
      .then((res) => setUnlocked(res.unlocked))
      .catch(() => {});
    refreshPacksCount();
  }, []);

  const handleToggleTheme = () => {
    setDark((prev) => !prev);
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
            {/* The ModelPacksPage/DocsGlossaryPage modules stay in src/ for
                reference, but are no longer routed: both legacy paths land on
                the Playground instead. */}
            <Route path="/packs" element={<Navigate to="/" replace />} />
            <Route path="/docs" element={<Navigate to="/" replace />} />
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
