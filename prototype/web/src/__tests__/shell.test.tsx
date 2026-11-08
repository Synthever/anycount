import { render, screen, fireEvent } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, it, expect, vi } from "vitest";
import { Sidebar } from "../components/Sidebar";
import { Topbar } from "../components/Topbar";
import { applyTheme, loadThemePreference, persistThemePreference } from "../App";

// Node 26's jsdom preset does not always expose Web Storage; provide a
// deterministic in-memory one for the persistence test.
const memoryStore = new Map<string, string>();
Object.defineProperty(window, "localStorage", {
  configurable: true,
  value: {
    getItem: (key: string) => memoryStore.get(key) ?? null,
    setItem: (key: string, value: string) => void memoryStore.set(key, String(value)),
    removeItem: (key: string) => void memoryStore.delete(key),
    clear: () => memoryStore.clear(),
    key: (index: number) => Array.from(memoryStore.keys())[index] ?? null,
    get length() {
      return memoryStore.size;
    },
  },
});

describe("AnyCount Studio Shell", () => {
  it("renders only the Playground and Benchmark destinations and toggles dark/cream theme", () => {
    document.documentElement.classList.add("dark");
    const onToggleTheme = vi.fn(() => {
      document.documentElement.classList.toggle("dark");
    });

    render(
      <MemoryRouter initialEntries={["/"]}>
        <Sidebar collapsed={false} packCount={1} />
        <Topbar
          pageTitle="Playground"
          dark={true}
          onToggleTheme={onToggleTheme}
          unlocked={false}
          onOpenUnlock={() => {}}
          onToggleSidebar={() => {}}
        />
      </MemoryRouter>
    );

    expect(screen.getByText("AnyCount Studio")).toBeTruthy();
    expect(screen.getAllByText("Playground").length).toBeGreaterThan(0);
    expect(screen.getByText("Benchmark Gallery")).toBeTruthy();
    expect(screen.queryByText("Model Packs (.onnx)")).toBeNull();
    expect(screen.queryByText("Method Docs & Glossary")).toBeNull();
    expect(screen.getByText("7 Counting Methods")).toBeTruthy();
    expect(screen.getByText("M1–M7")).toBeTruthy();

    const themeBtn = screen.getByRole("button", { name: /theme/i });
    fireEvent.click(themeBtn);
    expect(onToggleTheme).toHaveBeenCalledTimes(1);
    expect(document.documentElement.classList.contains("dark")).toBe(false);
  });

  it("persists the theme choice so a reload keeps Warm Cream", () => {
    window.localStorage.clear();

    applyTheme(false);
    persistThemePreference(false);
    expect(document.documentElement.classList.contains("light-cream")).toBe(true);
    expect(document.documentElement.classList.contains("dark")).toBe(false);
    expect(loadThemePreference()).toBe(false);

    applyTheme(true);
    persistThemePreference(true);
    expect(document.documentElement.classList.contains("dark")).toBe(true);
    expect(loadThemePreference()).toBe(true);
  });
});
