import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          {/* Pages will be wired in Task 12–16 */}
          <Route path="/" element={<Navigate to="/teach" replace />} />
          <Route
            path="*"
            element={
              <div className="flex h-screen items-center justify-center">
                <p className="text-muted-foreground">MindForge — scaffolding ready</p>
              </div>
            }
          />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
