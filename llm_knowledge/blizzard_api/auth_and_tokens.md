Client Credentials Flow
The OAuth client credentials flow is used to exchange a pair of client credentials (client_id and client_secret) for an access token.

See the OAuth RFC for detailed information about the client credentials flow.

The client credentials flow is used for most API requests, with the exception of those listed on the OAuth authorization code flow page.

Retrieving an access token
To request access tokens, an application must make a POST request with the following multipart form data to the token URI: grant_type=client_credentials

The application must pass basic HTTP auth credentials using the client_id as the user and client_secret as the password.

Example CURL request
curl -u {client_id}:{client_secret} -d grant_type=client_credentials https://oauth.battle.net/token
Expected response
{"access_token": "USVb1nGO9kwQlhNRRnI4iWVy2UV5j7M6h7", "token_type": "bearer", "expires_in": 86399, "scope": "example.scope"}
Using an access token
After an application retrieves an access token, it provides that token when making requests to API resources. This is done via an authorization header:

Example CURL reqest
curl --header "Authorization: Bearer <access_token>" <REST API URL>
Example use
Authorization: Bearer {token}


Application Authentication
POST
Access Token Request
/token
This is the only request necessary for the client credential flow, OAuth's authentication flow intended for application servers.

Parameter	Type	Example Value	Description
:region	string
Required
us	
The region of the data to retrieve.

grant_type	string
Required
client_credentials	
Identifies the type of authorization request being made. For a client credentials grant this value must be client_credentials.

scope	string		
A space-delimited, case-sensitive list of scopes that to which to request access. The user may not grant access to any or all requested scopes. See Using OAuth for a list of valid scopes.

Token Validation
POST
Token Validation (POST)
/check_token
Verifies that a given bearer token is valid and retrieves metadata about the token, including the client_id used to create the token, expiration timestamp, and scopes granted to the token. We strongly recommend that developers use the more secure POST /check_token method.

Parameter	Type	Example Value	Description
:region	string
Required
us	
The region of the data to retrieve.

token	string
Required
{token}	
The user's bearer token.

GET
Token Validation (GET)
/check_token
Verifies that a given bearer token is valid and retrieves metadata about the token, including the client_id used to create the token, expiration timestamp, and scopes granted to the token. We strongly recommend that developers use the more secure POST /check_token method.

Parameter	Type	Example Value	Description
:region	string
Required
us	
The region of the data to retrieve.

token	string
Required
{token}	
The user's bearer token.

