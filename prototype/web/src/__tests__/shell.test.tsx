import { render, screen, fireEvent } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, it, expect, vi } from "vitest";
import { Sidebar } from "../components/Sidebar";
import { Topbar } from "../components/Topbar";

describe("AnyCount Studio Shell", () => {
  it("renders all 4 navigation destinations and toggles dark/cream theme", () => {
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
    expect(screen.getByText("Model Packs (.onnx)")).toBeTruthy();
    expect(screen.getByText("Method Docs & Glossary")).toBeTruthy();

    const themeBtn = screen.getByRole("button", { name: /theme/i });
    fireEvent.click(themeBtn);
    expect(onToggleTheme).toHaveBeenCalledTimes(1);
    expect(document.documentElement.classList.contains("dark")).toBe(false);
  });
});
