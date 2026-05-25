"""
数字人对话接口与知识图谱集成模块
实现数字人与Graph RAG系统的无缝对接
"""

import asyncio
import json
from typing import List, Dict, Optional, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
import logging
from enum import Enum

# 假设的依赖
from graph_rag_engine import GraphRAGEngine, RetrievalResult


class DialogueState(Enum):
    """对话状态"""
    GREETING = "greeting"
    QUERYING = "querying"
    CLARIFYING = "clarifying"
    RECOMMENDING = "recommending"
    PLANNING = "planning"
    CLOSING = "closing"


@dataclass
class DialogueContext:
    """对话上下文"""
    session_id: str
    user_id: str
    state: DialogueState
    history: List[Dict] = field(default_factory=list)
    current_intent: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    user_profile: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


@dataclass
class DigitalHumanResponse:
    """数字人响应"""
    text: str
    emotion: str  # happy, neutral, thinking, excited
    action: Optional[str] = None  # recommend, plan, show_map
    data: Optional[Dict] = None
    suggestions: List[str] = field(default_factory=list)


class UserProfileManager:
    """用户画像管理器"""
    
    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver
        self.logger = logging.getLogger(__name__)
    
    async def get_or_create_profile(self, user_id: str) -> Dict:
        """获取或创建用户画像"""
        with self.driver.session() as session:
            # 查询现有画像
            query = """
                MERGE (u:TouristProfile {id: $user_id})
                ON CREATE SET u.created_at = datetime(),
                              u.preferences = [],
                              u.history = []
                RETURN u
            """
            
            result = session.run(query, user_id=user_id)
            record = result.single()
            
            if record:
                return dict(record["u"])
            
            return {}
    
    async def update_profile(
        self,
        user_id: str,
        interaction_data: Dict
    ):
        """更新用户画像"""
        with self.driver.session() as session:
            # 更新偏好和历史
            query = """
                MATCH (u:TouristProfile {id: $user_id})
                SET u.preferences = CASE
                    WHEN $preferences IS NOT NULL 
                    THEN u.preferences + $preferences
                    ELSE u.preferences
                END,
                u.history = u.history + $history,
                u.updated_at = datetime()
            """
            
            session.run(
                query,
                user_id=user_id,
                preferences=interaction_data.get("preferences"),
                history=interaction_data.get("history")
            )


class DialogueStateManager:
    """对话状态管理器"""
    
    def __init__(self):
        self.contexts: Dict[str, DialogueContext] = {}
        self.logger = logging.getLogger(__name__)
    
    def get_context(self, session_id: str) -> Optional[DialogueContext]:
        """获取对话上下文"""
        return self.contexts.get(session_id)
    
    def create_context(
        self,
        session_id: str,
        user_id: str
    ) -> DialogueContext:
        """创建对话上下文"""
        context = DialogueContext(
            session_id=session_id,
            user_id=user_id,
            state=DialogueState.GREETING,
            history=[],
            entities=[],
            user_profile={},
            metadata={}
        )
        
        self.contexts[session_id] = context
        return context
    
    def update_context(
        self,
        session_id: str,
        state: DialogueState = None,
        intent: str = None,
        entities: List[str] = None
    ):
        """更新对话上下文"""
        context = self.contexts.get(session_id)
        if not context:
            return
        
        if state:
            context.state = state
        if intent:
            context.current_intent = intent
        if entities:
            context.entities = entities
    
    def add_to_history(
        self,
        session_id: str,
        role: str,
        content: str
    ):
        """添加对话历史"""
        context = self.contexts.get(session_id)
        if context:
            context.history.append({
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            })


class ResponseGenerator:
    """响应生成器"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
    
    async def generate_greeting(self, user_name: str = None) -> DigitalHumanResponse:
        """生成问候语"""
        if user_name:
            text = f"您好，{user_name}！我是您的文旅助手小游。我可以为您推荐景点、规划行程、介绍文化历史。请问有什么可以帮您的吗？"
        else:
            text = "您好！我是您的文旅助手小游。我可以为您推荐景点、规划行程、介绍文化历史。请问有什么可以帮您的吗？"
        
        return DigitalHumanResponse(
            text=text,
            emotion="happy",
            suggestions=[
                "推荐一些好玩的景点",
                "帮我规划一日游行程",
                "介绍一下故宫的历史"
            ]
        )
    
    async def generate_response_from_results(
        self,
        query: str,
        results: List[RetrievalResult],
        context: DialogueContext
    ) -> DigitalHumanResponse:
        """基于检索结果生成响应"""
        # 构建上下文
        context_text = self._build_context_text(results)
        
        # 构建提示词
        prompt = self._build_prompt(query, context_text, context)
        
        try:
            # 调用LLM生成响应
            response = await self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个友好的文旅助手，名叫小游。请基于提供的知识图谱信息回答用户问题，保持专业且亲切的语气。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            
            # 生成建议问题
            suggestions = await self._generate_suggestions(query, results)
            
            # 确定情绪和动作
            emotion, action = self._determine_emotion_and_action(results)
            
            return DigitalHumanResponse(
                text=answer,
                emotion=emotion,
                action=action,
                data={"results": [self._result_to_dict(r) for r in results[:3]]},
                suggestions=suggestions
            )
        
        except Exception as e:
            self.logger.error(f"生成响应失败: {e}")
            return DigitalHumanResponse(
                text="抱歉，我遇到了一些问题。请稍后再试。",
                emotion="neutral"
            )
    
    def _build_context_text(self, results: List[RetrievalResult]) -> str:
        """构建上下文文本"""
        context_parts = []
        
        for i, result in enumerate(results[:5], 1):
            props = result.properties
            
            # 根据实体类型格式化信息
            if result.entity_type == "ScenicSpot":
                context_parts.append(
                    f"{i}. {result.name}\n"
                    f"   等级: {props.get('level', '未知')}\n"
                    f"   评分: {props.get('rating', '暂无')}\n"
                    f"   介绍: {props.get('description', '暂无介绍')[:100]}...\n"
                )
            elif result.entity_type == "TravelRoute":
                context_parts.append(
                    f"{i}. {result.name}\n"
                    f"   时长: {props.get('duration', '未知')}天\n"
                    f"   主题: {props.get('theme', '综合')}\n"
                )
            else:
                context_parts.append(
                    f"{i}. {result.name} ({result.entity_type})\n"
                )
        
        return "\n".join(context_parts)
    
    def _build_prompt(
        self,
        query: str,
        context_text: str,
        dialogue_context: DialogueContext
    ) -> str:
        """构建提示词"""
        # 包含对话历史
        history_text = ""
        if dialogue_context.history:
            recent_history = dialogue_context.history[-3:]  # 最近3轮对话
            history_text = "\n".join([
                f"{'用户' if h['role'] == 'user' else '助手'}: {h['content']}"
                for h in recent_history
            ])
        
        prompt = f"""
        用户问题: {query}
        
        相关知识图谱信息:
        {context_text}
        
        对话历史:
        {history_text}
        
        请基于以上信息回答用户问题，要求：
        1. 回答准确、详细
        2. 语气亲切、专业
        3. 适当引用知识图谱中的信息
        4. 如果信息不足，诚实告知
        """
        
        return prompt
    
    async def _generate_suggestions(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> List[str]:
        """生成建议问题"""
        suggestions = []
        
        # 基于结果类型生成建议
        if results and results[0].entity_type == "ScenicSpot":
            spot_name = results[0].name
            suggestions = [
                f"{spot_name}的门票多少钱",
                f"怎么去{spot_name}",
                f"{spot_name}附近有什么好吃的"
            ]
        elif results and results[0].entity_type == "TravelRoute":
            suggestions = [
                "这条路线适合什么季节",
                "路线的总预算是多少",
                "可以调整行程吗"
            ]
        else:
            suggestions = [
                "推荐一些热门景点",
                "帮我规划行程",
                "介绍一下当地文化"
            ]
        
        return suggestions[:3]
    
    def _determine_emotion_and_action(
        self,
        results: List[RetrievalResult]
    ) -> Tuple[str, Optional[str]]:
        """确定情绪和动作"""
        if not results:
            return "neutral", None
        
        top_result = results[0]
        
        if top_result.entity_type == "ScenicSpot":
            return "excited", "recommend"
        elif top_result.entity_type == "TravelRoute":
            return "happy", "plan"
        else:
            return "happy", None
    
    def _result_to_dict(self, result: RetrievalResult) -> Dict:
        """将检索结果转为字典"""
        return {
            "id": result.entity_id,
            "name": result.name,
            "type": result.entity_type,
            "score": result.score,
            "properties": result.properties
        }


class DigitalHumanDialogueSystem:
    """数字人对话系统"""
    
    def __init__(
        self,
        graph_rag_engine: GraphRAGEngine,
        neo4j_driver,
        llm_client
    ):
        self.graph_rag_engine = graph_rag_engine
        self.user_profile_manager = UserProfileManager(neo4j_driver)
        self.state_manager = DialogueStateManager()
        self.response_generator = ResponseGenerator(llm_client)
        
        self.logger = logging.getLogger(__name__)
    
    async def start_session(
        self,
        session_id: str,
        user_id: str
    ) -> DigitalHumanResponse:
        """开始对话会话"""
        # 创建对话上下文
        context = self.state_manager.create_context(session_id, user_id)
        
        # 获取用户画像
        user_profile = await self.user_profile_manager.get_or_create_profile(user_id)
        context.user_profile = user_profile
        
        # 生成问候语
        user_name = user_profile.get("name")
        greeting = await self.response_generator.generate_greeting(user_name)
        
        # 记录到历史
        self.state_manager.add_to_history(
            session_id,
            "assistant",
            greeting.text
        )
        
        return greeting
    
    async def process_message(
        self,
        session_id: str,
        user_message: str
    ) -> DigitalHumanResponse:
        """处理用户消息"""
        # 获取对话上下文
        context = self.state_manager.get_context(session_id)
        if not context:
            # 如果没有上下文，创建新会话
            return await self.start_session(session_id, "anonymous")
        
        # 记录用户消息
        self.state_manager.add_to_history(session_id, "user", user_message)
        
        try:
            # 1. 使用Graph RAG检索相关信息
            retrieval_results = await self.graph_rag_engine.retrieve(
                query=user_message,
                top_k=5,
                context={
                    "history": context.history,
                    "user_profile": context.user_profile
                }
            )
            
            # 2. 生成响应
            response = await self.response_generator.generate_response_from_results(
                user_message,
                retrieval_results,
                context
            )
            
            # 3. 更新对话状态
            self._update_dialogue_state(context, retrieval_results)
            
            # 4. 更新用户画像
            await self._update_user_interaction(
                context.user_id,
                user_message,
                retrieval_results
            )
            
            # 5. 记录助手响应
            self.state_manager.add_to_history(
                session_id,
                "assistant",
                response.text
            )
            
            return response
        
        except Exception as e:
            self.logger.error(f"处理消息失败: {e}")
            return DigitalHumanResponse(
                text="抱歉，我遇到了一些问题。请稍后再试。",
                emotion="neutral"
            )
    
    def _update_dialogue_state(
        self,
        context: DialogueContext,
        results: List[RetrievalResult]
    ):
        """更新对话状态"""
        if not results:
            context.state = DialogueState.QUERYING
            return
        
        top_result = results[0]
        
        if top_result.entity_type == "ScenicSpot":
            context.state = DialogueState.RECOMMENDING
        elif top_result.entity_type == "TravelRoute":
            context.state = DialogueState.PLANNING
        else:
            context.state = DialogueState.QUERYING
    
    async def _update_user_interaction(
        self,
        user_id: str,
        query: str,
        results: List[RetrievalResult]
    ):
        """更新用户交互记录"""
        # 提取偏好
        preferences = []
        for result in results[:3]:
            if result.entity_type == "ScenicSpot":
                category = result.properties.get("category")
                if category:
                    preferences.extend(category if isinstance(category, list) else [category])
        
        # 记录历史
        history_entry = {
            "query": query,
            "results": [r.name for r in results[:3]],
            "timestamp": datetime.now().isoformat()
        }
        
        # 更新画像
        await self.user_profile_manager.update_profile(
            user_id,
            {
                "preferences": list(set(preferences)),
                "history": [history_entry]
            }
        )
    
    async def get_personalized_recommendations(
        self,
        user_id: str,
        limit: int = 5
    ) -> List[RetrievalResult]:
        """获取个性化推荐"""
        # 获取用户画像
        profile = await self.user_profile_manager.get_or_create_profile(user_id)
        
        # 基于用户偏好构建查询
        preferences = profile.get("preferences", [])
        if preferences:
            query = f"推荐一些{', '.join(preferences[:3])}类型的景点"
        else:
            query = "推荐一些热门景点"
        
        # 使用Graph RAG检索
        results = await self.graph_rag_engine.retrieve(query, top_k=limit)
        
        return results
    
    async def end_session(self, session_id: str):
        """结束对话会话"""
        context = self.state_manager.get_context(session_id)
        if context:
            # 保存会话摘要
            self.logger.info(f"会话结束: {session_id}, 对话轮数: {len(context.history)}")
            
            # 删除上下文
            del self.state_manager.contexts[session_id]


class DigitalHumanAPI:
    """数字人API接口"""
    
    def __init__(self, dialogue_system: DigitalHumanDialogueSystem):
        self.dialogue_system = dialogue_system
        self.logger = logging.getLogger(__name__)
    
    async def handle_request(
        self,
        request: Dict
    ) -> Dict:
        """处理API请求"""
        action = request.get("action")
        session_id = request.get("session_id")
        user_id = request.get("user_id", "anonymous")
        
        try:
            if action == "start":
                response = await self.dialogue_system.start_session(
                    session_id,
                    user_id
                )
            
            elif action == "message":
                user_message = request.get("message", "")
                response = await self.dialogue_system.process_message(
                    session_id,
                    user_message
                )
            
            elif action == "recommend":
                response_results = await self.dialogue_system.get_personalized_recommendations(
                    user_id
                )
                response = DigitalHumanResponse(
                    text="为您推荐以下景点：",
                    emotion="happy",
                    action="recommend",
                    data={"results": [self._result_to_dict(r) for r in response_results]}
                )
            
            elif action == "end":
                await self.dialogue_system.end_session(session_id)
                response = DigitalHumanResponse(
                    text="感谢您的使用，祝您旅途愉快！",
                    emotion="happy"
                )
            
            else:
                response = DigitalHumanResponse(
                    text="未知的操作类型",
                    emotion="neutral"
                )
            
            return self._response_to_dict(response)
        
        except Exception as e:
            self.logger.error(f"处理请求失败: {e}")
            return {
                "text": "服务暂时不可用，请稍后再试",
                "emotion": "neutral",
                "error": str(e)
            }
    
    def _response_to_dict(self, response: DigitalHumanResponse) -> Dict:
        """将响应转为字典"""
        return {
            "text": response.text,
            "emotion": response.emotion,
            "action": response.action,
            "data": response.data,
            "suggestions": response.suggestions
        }
    
    def _result_to_dict(self, result: RetrievalResult) -> Dict:
        """将检索结果转为字典"""
        return {
            "id": result.entity_id,
            "name": result.name,
            "type": result.entity_type,
            "score": result.score,
            "properties": result.properties
        }


# 使用示例
async def main():
    """示例：使用数字人对话系统"""
    
    # 初始化系统（需要实际的连接）
    # dialogue_system = DigitalHumanDialogueSystem(
    #     graph_rag_engine,
    #     neo4j_driver,
    #     llm_client
    # )
    
    # api = DigitalHumanAPI(dialogue_system)
    
    # # 开始会话
    # start_response = await api.handle_request({
    #     "action": "start",
    #     "session_id": "session_001",
    #     "user_id": "user_001"
    # })
    
    # # 发送消息
    # message_response = await api.handle_request({
    #     "action": "message",
    #     "session_id": "session_001",
    #     "message": "推荐一些北京的历史景点"
    # })
    
    print("数字人对话系统示例")


if __name__ == "__main__":
    asyncio.run(main())
