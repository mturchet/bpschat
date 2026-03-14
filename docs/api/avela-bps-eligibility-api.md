# Avela / BPS Eligibility API — Address, ZIP + Grade → Schools

This document describes how to obtain **eligible BPS schools** from **address or ZIP** and **grade**. It covers the relationship between **boston.explore.avela.org** and the **documented** programmatic API used for eligibility.

---

## 1. boston.explore.avela.org vs programmatic API

| What | Description |
|------|--------------|
| **boston.explore.avela.org** | Public **web UI** for the Avela Explore school finder for Boston. Families use it to discover and compare BPS schools. It is a website, not an API base URL. |
| **Programmatic API for eligibility** | The **documented** API that returns eligible schools for a given address and grade is the **BPS Discover Service** at **api.mybps.org** (see below). It is operated by Boston Public Schools. The Explore site may use this API or an equivalent backend. |

**For the MVP chatbot:** Use the **BPS Discover Service** at `api.mybps.org` to implement “address/ZIP + grade → eligible schools.” No separate Avela API base URL is required unless you need an Avela-hosted backend (see §4).

---

## 2. BPS Discover Service (api.mybps.org)

**Base URL:**  
`http://api.mybps.org/BPSDiscoverService/Schools.svc`

**Service help (list of operations):**  
`http://api.mybps.org/BPSDiscoverService/Schools.svc/help`

### 2.1 Authentication

- **Auth:** None documented. Endpoints are public GET requests; no API key or OAuth is shown in the service help.
- **Protocol:** HTTP (consider HTTPS if/when available for production).

### 2.2 Flow: address/ZIP + grade → schools

Use two steps:

1. **Address lookup** → get `AddressID` (and optionally validate that the address is in Boston).
2. **Eligible schools** → call HomeSchools (or ZoneSchools) with `AddressID` and `Grade`.

---

## 3. Endpoints

### 3.1 Address lookup: AddressMatches

Resolves a street address (or address components) to one or more matches. Each match includes an **AddressID** required for HomeSchools/ZoneSchools.

**URL:**  
`GET http://api.mybps.org/BPSDiscoverService/Schools.svc/AddressMatches`

**Query parameters:**

| Parameter      | Description                          |
|----------------|--------------------------------------|
| `StreetNumber` | Street number (e.g. `"100"`)         |
| `Street`       | Street name (e.g. `"Warren St"`)     |
| `ZipCode`      | ZIP code (e.g. `"02119"`)            |

**Example:**  
`http://api.mybps.org/BPSDiscoverService/Schools.svc/AddressMatches?StreetNumber=100&Street=Warren%20St&ZipCode=02119`

**Response (JSON):**  
- `Error`: array of `{ "ID", "Message", "Method" }` (if any).
- `List`: array of address objects, each with:
  - `AddressID` — **use this for HomeSchools/ZoneSchools**
  - `Street`, `StreetNum`, `ZipCode`
  - `Lat`, `Lng`, `GeoCode`, `Zone`, `SectionOfCity`
  - `ELLCluster`, `SPEDCluster`
  - `X`, `Y`

If `List` is empty or `Error` is non-empty, the address was not found or invalid; do not call HomeSchools.

**ZIP-only usage:**  
The API expects street number + street + ZIP. For ZIP-only flows, you may need to try a placeholder or check whether the service accepts partial input; otherwise, the chatbot should ask for full address when the API requires it.

---

### 3.2 Eligible schools: HomeSchools

Returns the list of **eligible (home) schools** for a given address and grade.

**URL:**  
`GET http://api.mybps.org/BPSDiscoverService/Schools.svc/HomeSchools`

**Query parameters:**

| Parameter        | Description                                      |
|------------------|--------------------------------------------------|
| `schyear`        | School year (e.g. `"2025"` or `"2024"`)          |
| `Grade`          | Grade level (e.g. K1–12; use BPS’s grade codes)  |
| `AddressID`      | From AddressMatches (required)                    |
| `SiblingSchList` | Optional; comma-separated school IDs if sibling already attends |
| `IsAwc`          | Optional; e.g. `true`/`false` (assignment/walk eligibility)       |

**Example:**  
`http://api.mybps.org/BPSDiscoverService/Schools.svc/HomeSchools?schyear=2025&Grade=3&AddressID=<AddressID>&SiblingSchList=&IsAwc=false`

**Response (JSON):**  
- `Error`: array of `{ "ID", "Message", "Method" }` (if any).
- `List`: array of **SchoolChoice** objects, e.g.:
  - `SchoolID`, `SchoolName`, `Grade`, `SchoolYear`
  - `Eligibility`, `Tier`, `DeseTier`, `ProgramCode`
  - `IsExamSchool`, `IsSpecAdmission`, `IsAwc`
  - `StraightLineDistance`, `WalkLineDistance`, `TransEligible`, `AssignmentWalkEligibilityStatus`
  - `latitude`, `longitude`, `X`, `Y`
  - `NumClasses`, `SortClause`

All displayed schools in the chatbot should come from this `List` (or ZoneSchools); do not invent school names or eligibility.

---

### 3.3 Zone schools (alternative): ZoneSchools

Returns **zone schools** for an address and grade. Same idea as HomeSchools but zone-based.

**URL:**  
`GET http://api.mybps.org/BPSDiscoverService/Schools.svc/ZoneSchools`

**Query parameters:**  
`schyear`, `Grade`, `AddressID`, `SiblingSchList` (no `IsAwc` in the documented signature).

**Response shape:** Same `ValidSchoolChoices` structure as HomeSchools (`List` of `SchoolChoice`, plus `Error`).

---

## 4. Request/response format

- **Method:** GET for all endpoints above.
- **Request body:** None (parameters are query only).
- **Response format:** Both **XML** and **JSON** are supported. Prefer **JSON** by sending `Accept: application/json` (or use the same format the service returns by default for your client).

---

## 5. Error handling

- **AddressMatches:** If `List` is empty or `Error` is non-empty, treat as “invalid or out-of-district address.” Do not call HomeSchools; inform the user and do not return fabricated schools.
- **HomeSchools / ZoneSchools:** If `Error` is non-empty, surface a generic error message and do not return schools. If `List` is empty, respond that no eligible schools were found for that address/grade.

Implement timeouts and handle network/API unavailability; never substitute made-up school data.

---

## 6. Auth summary for “Avela API (boston.explore.avela.org)”

| System | Endpoint / URL | Auth |
|--------|----------------|------|
| **boston.explore.avela.org** | Web UI only (no API base URL) | N/A |
| **BPS Discover Service** (address + grade → schools) | `http://api.mybps.org/BPSDiscoverService/Schools.svc` | **None** (public GET) |
| **Avela Education Platform API** (avela.org/api) | Different product (applicants, enrollment); OAuth2, credentials from Avela | Contact info@avela.org |

For **address/ZIP + grade → eligible BPS schools**, the documented approach is the **BPS Discover Service** with **no authentication**. If the project later requires an Avela-hosted API (e.g. under boston.explore.avela.org), contact Avela (info@avela.org) for endpoint, parameters, and auth.

---

## 7. References

- BPS Discover Service help: `http://api.mybps.org/BPSDiscoverService/Schools.svc/help`
- AddressMatches operation: `http://api.mybps.org/BPSDiscoverService/Schools.svc/help/operations/GetAddressMatch`
- HomeSchools operation: `http://api.mybps.org/BPSDiscoverService/Schools.svc/help/operations/GetSC`
- ZoneSchools operation: `http://api.mybps.org/BPSDiscoverService/Schools.svc/help/operations/GetILE`
- Avela Explore (product): https://avela.org/explore
- BPS enrollment info: https://www.bostonpublicschools.org/enrollment
