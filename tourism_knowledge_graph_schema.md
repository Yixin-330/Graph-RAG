# 文旅场景知识图谱Schema设计文档

## 一、核心实体类型定义

### 1. 景区景点 (ScenicSpot)
```cypher
CREATE (s:ScenicSpot {
  id: String,              // 唯一标识
  name: String,            // 景点名称
  full_name: String,       // 完整名称
  level: String,           // 景区等级(5A/4A/3A)
  category: [String],      // 类型标签(自然/人文/主题公园等)
  description: String,     // 详细描述
  location: {              // 地理位置
    province: String,
    city: String,
    district: String,
    address: String,
    longitude: Float,
    latitude: Float
  },
  ticket: {                // 门票信息
    price: Float,
    discount: String,
    booking_url: String
  },
  open_time: String,       // 开放时间
  best_season: [String],   // 最佳游览季节
  duration: String,        // 建议游玩时长
  rating: Float,           // 评分
  popularity: Integer      // 热度指数
})
```

### 2. 文化遗产 (CulturalHeritage)
```cypher
CREATE (c:CulturalHeritage {
  id: String,
  name: String,
  heritage_type: String,   // 类型(物质/非物质文化遗产)
  protection_level: String, // 保护级别(世界/国家/省级)
  era: String,             // 所属年代
  history: String,         // 历史背景
  cultural_value: String,  // 文化价值描述
  location: {
    province: String,
    city: String,
    address: String
  },
  inheritors: [String],    // 传承人
  status: String           // 保护状态
})
```

### 3. 旅游路线 (TravelRoute)
```cypher
CREATE (r:TravelRoute {
  id: String,
  name: String,
  theme: String,           // 主题(历史文化/自然风光/美食之旅等)
  duration: Integer,       // 行程天数
  difficulty: String,      // 难度等级
  budget: {                // 预算范围
    min: Float,
    max: Float
  },
  transportation: [String], // 交通方式
  highlights: [String],    // 行程亮点
  itinerary: [             // 详细行程
    {
      day: Integer,
      spots: [String],
      activities: [String],
      accommodation: String
    }
  ],
  suitable_crowd: [String] // 适用人群
})
```

### 4. 服务设施 (ServiceFacility)
```cypher
CREATE (f:ServiceFacility {
  id: String,
  name: String,
  facility_type: String,   // 类型(餐饮/住宿/交通/购物/娱乐)
  category: String,        // 细分类别
  location: {
    province: String,
    city: String,
    address: String,
    longitude: Float,
    latitude: Float
  },
  price_level: String,     // 价格档次
  rating: Float,
  features: [String],      // 特色标签
  contact: {
    phone: String,
    website: String
  },
  opening_hours: String
})
```

### 5. 活动节庆 (Event)
```cypher
CREATE (e:Event {
  id: String,
  name: String,
  event_type: String,      // 类型(节庆/展览/演出/赛事)
  start_date: Date,
  end_date: Date,
  location: {
    province: String,
    city: String,
    venue: String
  },
  description: String,
  highlights: [String],
  ticket_info: String,
  organizer: String,
  scale: String            // 规模等级
})
```

### 6. 游客画像 (TouristProfile)
```cypher
CREATE (t:TouristProfile {
  id: String,
  age_group: String,       // 年龄段
  travel_preference: [String], // 旅游偏好
  budget_range: {
    min: Float,
    max: Float
  },
  travel_frequency: String, // 出游频率
  interests: [String],     // 兴趣标签
  history: [               // 历史行为
    {
      spot_id: String,
      visit_time: Date,
      rating: Float,
      review: String
    }
  ],
  constraints: {           // 限制条件
    mobility: String,      // 行动能力
    dietary: [String]      // 饮食限制
  }
})
```

## 二、核心关系类型定义

### 1. 空间关系
```cypher
// 景区包含景点
(s1:ScenicSpot)-[:CONTAINS {level: Integer}]->(s2:ScenicSpot)

// 景点邻近关系
(s1:ScenicSpot)-[:NEARBY {
  distance: Float,
  walk_time: Integer,
  transport_time: Integer
}]->(s2:ScenicSpot)

// 位于行政区划
(s:ScenicSpot)-[:LOCATED_IN]->(a:AdministrativeArea)
```

### 2. 旅游关系
```cypher
// 路线经过景点
(r:TravelRoute)-[:PASSES_THROUGH {
  day: Integer,
  order: Integer,
  duration: String
}]->(s:ScenicSpot)

// 游客游览景点
(t:TouristProfile)-[:VISITED {
  time: Date,
  rating: Float,
  duration: String
}]->(s:ScenicSpot)

// 景点推荐给游客
(s:ScenicSpot)-[:RECOMMENDED_TO {
  score: Float,
  reason: String
}]->(t:TouristProfile)
```

### 3. 文化关系
```cypher
// 文化遗产关联景点
(c:CulturalHeritage)-[:RELATED_TO {
  relation_type: String,
  description: String
}]->(s:ScenicSpot)

// 活动举办于景区
(e:Event)-[:HELD_AT]->(s:ScenicSpot)

// 景点文化主题
(s:ScenicSpot)-[:HAS_THEME]->(t:Theme)
```

### 4. 服务关系
```cypher
// 设施服务景点
(f:ServiceFacility)-[:SERVES {
  distance: Float,
  service_type: String
}]->(s:ScenicSpot)

// 路线包含住宿
(r:TravelRoute)-[:INCLUDES_ACCOMMODATION {
  day: Integer
}]->(f:ServiceFacility)
```

### 5. 相似关系
```cypher
// 景点相似性
(s1:ScenicSpot)-[:SIMILAR_TO {
  similarity: Float,
  common_features: [String]
}]->(s2:ScenicSpot)

// 路线相似性
(r1:TravelRoute)-[:SIMILAR_TO {
  similarity: Float
}]->(r2:TravelRoute)
```

## 三、属性图索引设计

### 1. 节点索引
```cypher
// 景点名称索引
CREATE INDEX scenic_spot_name IF NOT EXISTS FOR (s:ScenicSpot) ON (s.name)

// 地理位置索引
CREATE INDEX scenic_spot_location IF NOT EXISTS FOR (s:ScenicSpot) ON (s.location.province, s.location.city)

// 景区等级索引
CREATE INDEX scenic_spot_level IF NOT EXISTS FOR (s:ScenicSpot) ON (s.level)

// 活动时间索引
CREATE INDEX event_date IF NOT EXISTS FOR (e:Event) ON (e.start_date, e.end_date)
```

### 2. 全文索引
```cypher
// 景点描述全文索引
CREATE FULLTEXT INDEX scenic_spot_fulltext IF NOT EXISTS FOR (s:ScenicSpot) ON EACH [s.name, s.description]

// 文化遗产全文索引
CREATE FULLTEXT INDEX heritage_fulltext IF NOT EXISTS FOR (c:CulturalHeritage) ON EACH [c.name, c.history, c.cultural_value]
```

### 3. 向量索引（用于Graph RAG）
```cypher
// 景点描述向量索引
CREATE VECTOR INDEX scenic_spot_vector IF NOT EXISTS FOR (s:ScenicSpot) ON s.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 1536,
  `vector.similarity_function`: 'cosine'
}}
```

## 四、典型查询模式

### 1. 智能推荐查询
```cypher
// 基于用户画像的景点推荐
MATCH (t:TouristProfile {id: $user_id})
MATCH (s:ScenicSpot)
WHERE s.category IN t.interests
AND s.location.province = t.preferred_province
WITH s, 
     vector.similarity.cosine(s.embedding, $query_vector) AS similarity
ORDER BY similarity DESC, s.rating DESC
LIMIT 10
RETURN s
```

### 2. 路线规划查询
```cypher
// 查找经过指定景点的路线
MATCH (r:TravelRoute)-[p:PASSES_THROUGH]->(s:ScenicSpot)
WHERE s.id IN $spot_ids
WITH r, count(DISTINCT s) AS matched_spots
ORDER BY matched_spots DESC
RETURN r
```

### 3. 知识推理查询
```cypher
// 查找文化关联景点
MATCH (c:CulturalHeritage)-[:RELATED_TO]->(s1:ScenicSpot)
WHERE c.era CONTAINS '唐代'
MATCH (s1)-[:NEARBY]->(s2:ScenicSpot)
WHERE s2.level = '5A'
RETURN c, s1, s2
```

## 五、数据质量约束

### 1. 必填字段验证
- ScenicSpot: id, name, location
- CulturalHeritage: id, name, heritage_type
- TravelRoute: id, name, duration

### 2. 关系完整性约束
- CONTAINS关系不能形成环
- NEARBY关系必须对称
- PASSES_THROUGH必须按day和order有序

### 3. 数据更新策略
- 热度数据：每日更新
- 评分数据：实时更新
- 基础信息：人工审核后更新
