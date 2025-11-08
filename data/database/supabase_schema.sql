-- ============================================
-- HOSPITALS TABLE
-- ============================================
CREATE TABLE hospitals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    region VARCHAR(100) NOT NULL,
    latitude DECIMAL(9, 6),
    longitude DECIMAL(9, 6),
    capacity_beds INTEGER NOT NULL,
    hospital_type VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_hospitals_region ON hospitals(region);
CREATE INDEX idx_hospitals_type ON hospitals(hospital_type);

-- ============================================
-- RESOURCE TYPES (Reference Table)
-- ============================================
CREATE TABLE resource_types (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    unit VARCHAR(20) NOT NULL,
    critical_threshold INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Insert resource types
INSERT INTO resource_types (name, unit, critical_threshold) VALUES
    ('ventilators', 'units', 2),
    ('o2_cylinders', 'cylinders', 10),
    ('beds', 'beds', 5),
    ('medications', 'doses', 100),
    ('ppe', 'sets', 50)
ON CONFLICT (name) DO UPDATE
SET unit = EXCLUDED.unit,
    critical_threshold = EXCLUDED.critical_threshold;

-- ============================================
-- RESOURCE INVENTORY (Current State)
-- ============================================
CREATE TABLE resource_inventory (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    hospital_id UUID NOT NULL REFERENCES hospitals(id) ON DELETE CASCADE,
    resource_type_id INTEGER NOT NULL REFERENCES resource_types(id),
    quantity INTEGER NOT NULL CHECK (quantity >= 0),
    reserved_quantity INTEGER DEFAULT 0 CHECK (reserved_quantity >= 0),
    available_quantity INTEGER GENERATED ALWAYS AS (quantity - reserved_quantity) STORED,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(hospital_id, resource_type_id)
);

CREATE INDEX idx_inventory_hospital ON resource_inventory(hospital_id);
CREATE INDEX idx_inventory_resource ON resource_inventory(resource_type_id);
CREATE INDEX idx_inventory_updated ON resource_inventory(last_updated);

-- ============================================
-- INVENTORY HISTORY (Time Series Data) - CORRECTED
-- ============================================
CREATE TABLE inventory_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    hospital_id UUID NOT NULL REFERENCES hospitals(id) ON DELETE CASCADE,
    resource_type_id INTEGER NOT NULL REFERENCES resource_types(id),
    quantity INTEGER NOT NULL,
    consumption INTEGER DEFAULT 0,        -- ADDED
    resupply INTEGER DEFAULT 0,           -- ADDED
    date DATE NOT NULL,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_history_hospital_resource ON inventory_history(hospital_id, resource_type_id, date);
CREATE INDEX idx_history_date ON inventory_history(date);

-- ============================================
-- PATIENT ADMISSIONS (For Demand Correlation)
-- ============================================
CREATE TABLE patient_admissions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    hospital_id UUID NOT NULL REFERENCES hospitals(id) ON DELETE CASCADE,
    admission_date DATE NOT NULL,
    total_admissions INTEGER NOT NULL CHECK (total_admissions >= 0),
    icu_admissions INTEGER DEFAULT 0 CHECK (icu_admissions >= 0),
    emergency_admissions INTEGER DEFAULT 0 CHECK (emergency_admissions >= 0),
    average_severity DECIMAL(3, 2) CHECK (average_severity BETWEEN 1.0 AND 5.0),
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(hospital_id, admission_date)
);

CREATE INDEX idx_admissions_hospital_date ON patient_admissions(hospital_id, admission_date);
CREATE INDEX idx_admissions_date ON patient_admissions(admission_date);

-- ============================================
-- RESOURCE REQUESTS
-- ============================================
CREATE TABLE resource_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    hospital_id UUID NOT NULL REFERENCES hospitals(id) ON DELETE CASCADE,
    resource_type_id INTEGER NOT NULL REFERENCES resource_types(id),
    requested_quantity INTEGER NOT NULL CHECK (requested_quantity > 0),
    urgency_level VARCHAR(20) NOT NULL CHECK (urgency_level IN ('low', 'medium', 'high', 'critical')),
    reason TEXT,
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'fulfilled', 'rejected')),
    requested_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_requests_hospital ON resource_requests(hospital_id);
CREATE INDEX idx_requests_status ON resource_requests(status);
CREATE INDEX idx_requests_urgency ON resource_requests(urgency_level);

-- ============================================
-- ALLOCATIONS (Transfer Records)
-- ============================================
CREATE TABLE allocations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    from_hospital_id UUID REFERENCES hospitals(id),
    to_hospital_id UUID NOT NULL REFERENCES hospitals(id),
    resource_type_id INTEGER NOT NULL REFERENCES resource_types(id),
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    transfer_cost DECIMAL(10, 2),
    estimated_time_hours INTEGER,
    status VARCHAR(20) DEFAULT 'proposed' CHECK (status IN ('proposed', 'approved', 'in_transit', 'completed', 'cancelled')),
    priority_level VARCHAR(20) CHECK (priority_level IN ('low', 'medium', 'high', 'critical')),
    allocation_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    notes TEXT,
    created_by VARCHAR(100),
    CONSTRAINT check_different_hospitals CHECK (from_hospital_id != to_hospital_id OR from_hospital_id IS NULL)
);

CREATE INDEX idx_allocations_from ON allocations(from_hospital_id);
CREATE INDEX idx_allocations_to ON allocations(to_hospital_id);
CREATE INDEX idx_allocations_status ON allocations(status);
CREATE INDEX idx_allocations_date ON allocations(allocation_date);

-- ============================================
-- USER INTERACTIONS (For Preference Learning)
-- ============================================
CREATE TABLE user_interactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(100),
    session_id VARCHAR(100),
    user_query TEXT NOT NULL,
    context JSONB,
    recommendations JSONB NOT NULL,
    selected_recommendation_index INTEGER,
    user_choice JSONB,
    explicit_feedback TEXT,
    feedback_rating INTEGER CHECK (feedback_rating BETWEEN 1 AND 5),
    framework_used VARCHAR(20) CHECK (framework_used IN ('langgraph', 'crewai')),
    response_time_seconds DECIMAL(6, 2),
    interaction_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_interactions_user ON user_interactions(user_id);
CREATE INDEX idx_interactions_session ON user_interactions(session_id);
CREATE INDEX idx_interactions_timestamp ON user_interactions(interaction_timestamp);
CREATE INDEX idx_interactions_framework ON user_interactions(framework_used);

-- ============================================
-- PREFERENCE PROFILES (Learned User Preferences)
-- ============================================
CREATE TABLE preference_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(100) NOT NULL UNIQUE,
    preference_weights JSONB NOT NULL,
    interaction_count INTEGER DEFAULT 0,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_preferences_user ON preference_profiles(user_id);

-- ============================================
-- EVENTS (Outbreak, Surge, Supply Disruption)
-- ============================================
CREATE TABLE events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(50) NOT NULL,
    event_name VARCHAR(255) NOT NULL,
    affected_region VARCHAR(100),
    affected_hospitals UUID[],
    start_date DATE NOT NULL,
    end_date DATE,
    severity VARCHAR(20) CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    impact_description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_events_type ON events(event_type);
CREATE INDEX idx_events_date_range ON events(start_date, end_date);
CREATE INDEX idx_events_region ON events(affected_region);

-- ============================================
-- SHORTAGE PREDICTIONS (ML Model Outputs)
-- ============================================
CREATE TABLE shortage_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    hospital_id UUID NOT NULL REFERENCES hospitals(id) ON DELETE CASCADE,
    resource_type_id INTEGER NOT NULL REFERENCES resource_types(id),
    prediction_date DATE NOT NULL,
    predicted_quantity INTEGER NOT NULL,
    confidence_score DECIMAL(5, 4) CHECK (confidence_score BETWEEN 0 AND 1),
    risk_level VARCHAR(20) CHECK (risk_level IN ('low', 'medium', 'high', 'critical')),
    days_until_shortage INTEGER,
    predicted_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_predictions_hospital_resource ON shortage_predictions(hospital_id, resource_type_id);
CREATE INDEX idx_predictions_date ON shortage_predictions(prediction_date);
CREATE INDEX idx_predictions_risk ON shortage_predictions(risk_level);

-- ============================================
-- HELPER FUNCTIONS
-- ============================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_hospitals_updated_at
    BEFORE UPDATE ON hospitals
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- VIEWS FOR COMMON QUERIES
-- ============================================
CREATE VIEW v_current_resource_status AS
SELECT 
    h.id as hospital_id,
    h.name as hospital_name,
    h.region,
    rt.name as resource_type,
    ri.quantity,
    ri.reserved_quantity,
    ri.available_quantity,
    rt.critical_threshold,
    CASE 
        WHEN ri.available_quantity <= rt.critical_threshold THEN 'critical'
        WHEN ri.available_quantity <= rt.critical_threshold * 2 THEN 'low'
        ELSE 'sufficient'
    END as stock_status,
    ri.last_updated
FROM hospitals h
CROSS JOIN resource_types rt
LEFT JOIN resource_inventory ri ON h.id = ri.hospital_id AND rt.id = ri.resource_type_id
ORDER BY h.region, h.name, rt.name;

CREATE VIEW v_regional_resource_summary AS
SELECT 
    h.region,
    rt.name as resource_type,
    COUNT(DISTINCT h.id) as total_hospitals,
    COALESCE(SUM(ri.quantity), 0) as total_quantity,
    COALESCE(SUM(ri.available_quantity), 0) as total_available,
    COALESCE(AVG(ri.quantity), 0) as avg_quantity_per_hospital
FROM hospitals h
CROSS JOIN resource_types rt
LEFT JOIN resource_inventory ri ON h.id = ri.hospital_id AND rt.id = ri.resource_type_id
GROUP BY h.region, rt.name
ORDER BY h.region, rt.name;