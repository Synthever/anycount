use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("Passcode diperlukan untuk menjalankan fitur live ini")]
    AuthRequired,
    #[error("Passcode tidak sesuai")]
    InvalidPasscode,
    #[error("{0}")]
    BadRequest(String),
    #[error("{0}")]
    NotFound(String),
    #[error("{0}")]
    Conflict(String),
    #[error("Kesalahan internal server")]
    Internal(#[from] anyhow::Error),
}

impl AppError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::AuthRequired => "AUTH_REQUIRED",
            Self::InvalidPasscode => "AUTH_INVALID_PASSCODE",
            Self::BadRequest(_) => "BAD_REQUEST",
            Self::NotFound(_) => "NOT_FOUND",
            Self::Conflict(_) => "CONFLICT",
            Self::Internal(_) => "INTERNAL_ERROR",
        }
    }

    pub fn status(&self) -> StatusCode {
        match self {
            Self::AuthRequired | Self::InvalidPasscode => StatusCode::UNAUTHORIZED,
            Self::BadRequest(_) => StatusCode::BAD_REQUEST,
            Self::NotFound(_) => StatusCode::NOT_FOUND,
            Self::Conflict(_) => StatusCode::CONFLICT,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        if let Self::Internal(ref err) = self {
            tracing::error!(error = ?err, "internal error");
        }
        let status = self.status();
        let body = json!({
            "error": {
                "code": self.code(),
                "message": self.to_string(),
            }
        });
        (status, Json(body)).into_response()
    }
}
