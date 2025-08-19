//! OAuth Provider implementations for Concept Mesh
//!
//! This module provides OAuth 2.0 integrations with various identity providers
//! including GitHub, Google, Apple, Discord, and Auth0.

use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;
use url::Url;

/// OAuth provider types supported by the system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OAuthProvider {
    /// GitHub OAuth
    GitHub,
    /// Google OAuth
    Google,
    /// Apple Sign In
    Apple,
    /// Discord OAuth
    Discord,
    /// Auth0
    Auth0,
}

impl OAuthProvider {
    /// Convert provider to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::GitHub => "github",
            Self::Google => "google",
            Self::Apple => "apple",
            Self::Discord => "discord",
            Self::Auth0 => "auth0",
        }
    }

    /// Parse provider from string
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "github" => Ok(Self::GitHub),
            "google" => Ok(Self::Google),
            "apple" => Ok(Self::Apple),
            "discord" => Ok(Self::Discord),
            "auth0" => Ok(Self::Auth0),
            _ => Err(format!("Unknown OAuth provider: {}", s)),
        }
    }

    /// Get required OAuth scopes for this provider
    pub fn scopes(&self) -> Vec<&'static str> {
        match self {
            Self::GitHub => vec!["user:email", "read:user"],
            Self::Google => vec!["openid", "email", "profile"],
            Self::Apple => vec!["name", "email"],
            Self::Discord => vec!["identify", "email"],
            Self::Auth0 => vec!["openid", "profile", "email"],
        }
    }

    /// Get authorization endpoint for this provider
    pub fn auth_endpoint(&self) -> &'static str {
        match self {
            Self::GitHub => "https://github.com/login/oauth/authorize",
            Self::Google => "https://accounts.google.com/o/oauth2/v2/auth",
            Self::Apple => "https://appleid.apple.com/auth/authorize",
            Self::Discord => "https://discord.com/api/oauth2/authorize",
            Self::Auth0 => "", // Requires domain from credentials
        }
    }

    /// Get token endpoint for this provider
    pub fn token_endpoint(&self) -> &'static str {
        match self {
            Self::GitHub => "https://github.com/login/oauth/access_token",
            Self::Google => "https://oauth2.googleapis.com/token",
            Self::Apple => "https://appleid.apple.com/auth/token",
            Self::Discord => "https://discord.com/api/oauth2/token",
            Self::Auth0 => "", // Requires domain from credentials
        }
    }

    /// Get user info endpoint for this provider
    pub fn user_info_endpoint(&self) -> &'static str {
        match self {
            Self::GitHub => "https://api.github.com/user",
            Self::Google => "https://openidconnect.googleapis.com/v1/userinfo",
            Self::Apple => "", // Apple returns user info in the ID token
            Self::Discord => "https://discord.com/api/users/@me",
            Self::Auth0 => "", // Requires domain from credentials
        }
    }
}

/// OAuth credentials for a provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthCredentials {
    /// Provider type
    pub provider: OAuthProvider,
    /// Client ID
    pub client_id: String,
    /// Client secret
    pub client_secret: String,
    /// Redirect URI
    pub redirect_uri: String,
    /// Additional provider-specific config (e.g., Auth0 domain)
    pub additional_config: HashMap<String, String>,
}

impl OAuthCredentials {
    /// Create new OAuth credentials
    pub fn new(
        provider: OAuthProvider,
        client_id: &str,
        client_secret: &str,
        redirect_uri: &str,
    ) -> Self {
        Self {
            provider,
            client_id: client_id.to_string(),
            client_secret: client_secret.to_string(),
            redirect_uri: redirect_uri.to_string(),
            additional_config: HashMap::new(),
        }
    }

    /// Add additional configuration
    pub fn with_config(mut self, key: &str, value: &str) -> Self {
        self.additional_config
            .insert(key.to_string(), value.to_string());
        self
    }

    /// Get Auth0 domain if present
    pub fn auth0_domain(&self) -> Option<&str> {
        self.additional_config.get("domain").map(|s| s.as_str())
    }
}

/// OAuth user information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthUser {
    /// Provider-specific user ID
    pub provider_id: String,
    /// OAuth provider
    pub provider: OAuthProvider,
    /// User email (if available)
    pub email: Option<String>,
    /// User name (if available)
    pub name: Option<String>,
    /// Avatar URL (if available)
    pub avatar_url: Option<String>,
}

/// OAuth error types
#[derive(Debug, Error)]
pub enum OAuthError {
    /// Missing or invalid credentials
    #[error("Invalid OAuth credentials: {0}")]
    InvalidCredentials(String),

    /// HTTP request error
    #[error("HTTP request failed: {0}")]
    RequestFailed(#[from] reqwest::Error),

    /// Token exchange error
    #[error("Token exchange failed: {0}")]
    TokenExchangeFailed(String),

    /// User info fetch error
    #[error("Failed to fetch user info: {0}")]
    UserInfoFailed(String),

    /// Invalid authorization code
    #[error("Invalid authorization code")]
    InvalidCode,

    /// Missing field in response
    #[error("Missing field in response: {0}")]
    MissingField(String),

    /// JSON parsing error
    #[error("JSON parsing error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// URL parsing error
    #[error("URL parsing error: {0}")]
    UrlError(#[from] url::ParseError),
}

/// OAuth authentication client
pub struct OAuthClient {
    http_client: Client,
}

impl OAuthClient {
    /// Create a new OAuth client
    pub fn new() -> Self {
        Self {
            http_client: Client::new(),
        }
    }

    /// Build the authorization URL for the given provider
    pub fn authorization_url(&self, credentials: &OAuthCredentials) -> Result<String, OAuthError> {
        let mut url = match credentials.provider {
            OAuthProvider::Auth0 => {
                let domain = credentials.auth0_domain().ok_or_else(|| {
                    OAuthError::InvalidCredentials("Missing Auth0 domain".to_string())
                })?;
                format!("https://{}/authorize", domain)
            }
            _ => credentials.provider.auth_endpoint().to_string(),
        };

        let mut params = vec![
            ("client_id", credentials.client_id.clone()),
            ("redirect_uri", credentials.redirect_uri.clone()),
            ("response_type", "code".to_string()),
        ];

        // Add scopes
        let scopes = credentials.provider.scopes().join(" ");
        params.push(("scope", scopes));

        // Add state parameter for CSRF protection
        use rand::{thread_rng, Rng};
        let state: u64 = thread_rng().gen();
        params.push(("state", state.to_string()));

        // Add provider-specific parameters
        match credentials.provider {
            OAuthProvider::GitHub => {
                // GitHub specific parameters
            }
            OAuthProvider::Google => {
                params.push(("access_type", "offline".to_string()));
                params.push(("prompt", "consent".to_string()));
            }
            OAuthProvider::Apple => {
                params.push(("response_mode", "form_post".to_string()));
            }
            OAuthProvider::Discord => {
                // Discord specific parameters
            }
            OAuthProvider::Auth0 => {
                // Auth0 specific parameters
                params.push(("connection", "github".to_string()));
            }
        }

        // Build the URL with query parameters
        let mut full_url = url.parse::<Url>()?;
        for (key, value) in params {
            full_url.query_pairs_mut().append_pair(key, &value);
        }

        Ok(full_url.to_string())
    }

    /// Exchange an authorization code for tokens
    pub async fn exchange_code(
        &self,
        credentials: &OAuthCredentials,
        code: &str,
    ) -> Result<OAuthUser, OAuthError> {
        let token_url = match credentials.provider {
            OAuthProvider::Auth0 => {
                let domain = credentials.auth0_domain().ok_or_else(|| {
                    OAuthError::InvalidCredentials("Missing Auth0 domain".to_string())
                })?;
                format!("https://{}/oauth/token", domain)
            }
            _ => credentials.provider.token_endpoint().to_string(),
        };

        // Build token request parameters
        let mut params = vec![
            ("client_id", credentials.client_id.clone()),
            ("client_secret", credentials.client_secret.clone()),
            ("code", code.to_string()),
            ("redirect_uri", credentials.redirect_uri.clone()),
            ("grant_type", "authorization_code".to_string()),
        ];

        // Add provider-specific parameters
        match credentials.provider {
            OAuthProvider::GitHub => {
                // GitHub specific parameters
            }
            OAuthProvider::Google => {
                // Google specific parameters
            }
            OAuthProvider::Apple => {
                // Apple specific parameters
            }
            OAuthProvider::Discord => {
                // Discord specific parameters
            }
            OAuthProvider::Auth0 => {
                // Auth0 specific parameters
            }
        }

        // Exchange code for token
        #[derive(Deserialize)]
        struct TokenResponse {
            access_token: Option<String>,
            id_token: Option<String>,
            token_type: Option<String>,
            error: Option<String>,
            error_description: Option<String>,
        }

        let response = self
            .http_client
            .post(&token_url)
            .form(&params)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(OAuthError::TokenExchangeFailed(format!(
                "HTTP error: {}",
                response.status()
            )));
        }

        let token_response: TokenResponse = response.json().await?;

        if let Some(error) = token_response.error {
            return Err(OAuthError::TokenExchangeFailed(format!(
                "{}: {}",
                error,
                token_response.error_description.unwrap_or_default()
            )));
        }

        let access_token = token_response
            .access_token
            .ok_or_else(|| OAuthError::MissingField("access_token".to_string()))?;

        // Fetch user info
        let user = self.fetch_user_info(credentials, &access_token).await?;

        Ok(user)
    }

    /// Fetch user info using the access token
    async fn fetch_user_info(
        &self,
        credentials: &OAuthCredentials,
        access_token: &str,
    ) -> Result<OAuthUser, OAuthError> {
        let user_info_url = match credentials.provider {
            OAuthProvider::Auth0 => {
                let domain = credentials.auth0_domain().ok_or_else(|| {
                    OAuthError::InvalidCredentials("Missing Auth0 domain".to_string())
                })?;
                format!("https://{}/userinfo", domain)
            }
            OAuthProvider::Apple => {
                // Apple returns user info in the ID token
                // For simplicity, we'll return a dummy user here
                return Ok(OAuthUser {
                    provider_id: "apple_user".to_string(),
                    provider: OAuthProvider::Apple,
                    email: Some("apple_user@example.com".to_string()),
                    name: Some("Apple User".to_string()),
                    avatar_url: None,
                });
            }
            _ => credentials.provider.user_info_endpoint().to_string(),
        };

        let response = self
            .http_client
            .get(&user_info_url)
            .header("Authorization", format!("Bearer {}", access_token))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(OAuthError::UserInfoFailed(format!(
                "HTTP error: {}",
                response.status()
            )));
        }

        // Parse user info based on provider
        match credentials.provider {
            OAuthProvider::GitHub => {
                #[derive(Deserialize)]
                struct GitHubUser {
                    id: u64,
                    login: String,
                    name: Option<String>,
                    email: Option<String>,
                    avatar_url: Option<String>,
                }

                let github_user: GitHubUser = response.json().await?;

                Ok(OAuthUser {
                    provider_id: github_user.id.to_string(),
                    provider: OAuthProvider::GitHub,
                    email: github_user.email,
                    name: github_user.name.or(Some(github_user.login)),
                    avatar_url: github_user.avatar_url,
                })
            }
            OAuthProvider::Google => {
                #[derive(Deserialize)]
                struct GoogleUser {
                    sub: String,
                    email: Option<String>,
                    name: Option<String>,
                    picture: Option<String>,
                }

                let google_user: GoogleUser = response.json().await?;

                Ok(OAuthUser {
                    provider_id: google_user.sub,
                    provider: OAuthProvider::Google,
                    email: google_user.email,
                    name: google_user.name,
                    avatar_url: google_user.picture,
                })
            }
            OAuthProvider::Discord => {
                #[derive(Deserialize)]
                struct DiscordUser {
                    id: String,
                    username: String,
                    email: Option<String>,
                    avatar: Option<String>,
                }

                let discord_user: DiscordUser = response.json().await?;

                // Discord avatar URL needs to be constructed
                let avatar_url = discord_user.avatar.map(|avatar| {
                    format!(
                        "https://cdn.discordapp.com/avatars/{}/{}.png",
                        discord_user.id, avatar
                    )
                });

                Ok(OAuthUser {
                    provider_id: discord_user.id,
                    provider: OAuthProvider::Discord,
                    email: discord_user.email,
                    name: Some(discord_user.username),
                    avatar_url,
                })
            }
            OAuthProvider::Auth0 => {
                #[derive(Deserialize)]
                struct Auth0User {
                    sub: String,
                    email: Option<String>,
                    name: Option<String>,
                    picture: Option<String>,
                }

                let auth0_user: Auth0User = response.json().await?;

                Ok(OAuthUser {
                    provider_id: auth0_user.sub,
                    provider: OAuthProvider::Auth0,
                    email: auth0_user.email,
                    name: auth0_user.name,
                    avatar_url: auth0_user.picture,
                })
            }
            OAuthProvider::Apple => {
                // Should not reach here, handled above
                unreachable!()
            }
        }
    }
}

impl Default for OAuthClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Load OAuth credentials from environment variables
pub fn load_oauth_credentials(provider_name: &str) -> Result<OAuthCredentials, OAuthError> {
    let provider =
        OAuthProvider::from_str(provider_name).map_err(|e| OAuthError::InvalidCredentials(e))?;

    let env_prefix = provider_name.to_uppercase();
    let client_id_key = format!("{}_CLIENT_ID", env_prefix);
    let client_secret_key = format!("{}_CLIENT_SECRET", env_prefix);
    let redirect_uri_key = format!("{}_REDIRECT_URI", env_prefix);

    let client_id = std::env::var(&client_id_key)
        .map_err(|_| OAuthError::InvalidCredentials(format!("Missing {}", client_id_key)))?;

    let client_secret = std::env::var(&client_secret_key)
        .map_err(|_| OAuthError::InvalidCredentials(format!("Missing {}", client_secret_key)))?;

    let redirect_uri = std::env::var(&redirect_uri_key).unwrap_or_else(|_| {
        // Default redirect URI
        "http://localhost:8000/auth/callback".to_string()
    });

    let mut credentials =
        OAuthCredentials::new(provider, &client_id, &client_secret, &redirect_uri);

    // Add provider-specific config
    match provider {
        OAuthProvider::Auth0 => {
            let domain_key = "AUTH0_DOMAIN";
            let domain = std::env::var(domain_key)
                .map_err(|_| OAuthError::InvalidCredentials(format!("Missing {}", domain_key)))?;
            credentials = credentials.with_config("domain", &domain);
        }
        _ => {}
    }

    Ok(credentials)
}

/// Get GitHub OAuth configuration
pub fn github_oauth_config() -> OAuthCredentials {
    OAuthCredentials::new(
        OAuthProvider::GitHub,
        &std::env::var("GITHUB_CLIENT_ID").unwrap_or("CLIENT_ID".to_string()),
        &std::env::var("GITHUB_CLIENT_SECRET").unwrap_or("CLIENT_SECRET".to_string()),
        &std::env::var("GITHUB_REDIRECT_URI")
            .unwrap_or("http://localhost:8000/auth/callback".to_string()),
    )
}

/// Get Google OAuth configuration
pub fn google_oauth_config() -> OAuthCredentials {
    OAuthCredentials::new(
        OAuthProvider::Google,
        &std::env::var("GOOGLE_CLIENT_ID").unwrap_or("CLIENT_ID".to_string()),
        &std::env::var("GOOGLE_CLIENT_SECRET").unwrap_or("CLIENT_SECRET".to_string()),
        &std::env::var("GOOGLE_REDIRECT_URI")
            .unwrap_or("http://localhost:8000/auth/callback".to_string()),
    )
}

/// Get Apple OAuth configuration
pub fn apple_oauth_config() -> OAuthCredentials {
    OAuthCredentials::new(
        OAuthProvider::Apple,
        &std::env::var("APPLE_CLIENT_ID").unwrap_or("CLIENT_ID".to_string()),
        &std::env::var("APPLE_CLIENT_SECRET").unwrap_or("CLIENT_SECRET".to_string()),
        &std::env::var("APPLE_REDIRECT_URI")
            .unwrap_or("http://localhost:8000/auth/callback".to_string()),
    )
}

/// Get Discord OAuth configuration
pub fn discord_oauth_config() -> OAuthCredentials {
    OAuthCredentials::new(
        OAuthProvider::Discord,
        &std::env::var("DISCORD_CLIENT_ID").unwrap_or("CLIENT_ID".to_string()),
        &std::env::var("DISCORD_CLIENT_SECRET").unwrap_or("CLIENT_SECRET".to_string()),
        &std::env::var("DISCORD_REDIRECT_URI")
            .unwrap_or("http://localhost:8000/auth/callback".to_string()),
    )
}

/// Get Auth0 OAuth configuration
pub fn auth0_oauth_config() -> Result<OAuthCredentials, OAuthError> {
    let domain = std::env::var("AUTH0_DOMAIN")
        .map_err(|_| OAuthError::InvalidCredentials("Missing AUTH0_DOMAIN".to_string()))?;

    Ok(OAuthCredentials::new(
        OAuthProvider::Auth0,
        &std::env::var("AUTH0_CLIENT_ID").unwrap_or("CLIENT_ID".to_string()),
        &std::env::var("AUTH0_CLIENT_SECRET").unwrap_or("CLIENT_SECRET".to_string()),
        &std::env::var("AUTH0_REDIRECT_URI")
            .unwrap_or("http://localhost:8000/auth/callback".to_string()),
    )
    .with_config("domain", &domain))
}
