
# AI 카피라이터 도입을 통한 개인화 마케팅 실현 (Realizing personalized marketing through the introduction of AI copywriters)

## 연구 내용
- 고객 관심사 정보를 활용하여 마케팅 성공률을 높일 수 있는 컨텐츠 연구
- Generation AI 컨텐츠 플랫폼을 활용하기 위한 input data 정제 방법 연구

## 연구 목적
- Topic 모델과 generation AI 융합을 통한 비즈니스 성과 창출
- 마케팅 성공률을 향상 시키기 위한 개인화 마케팅 메시지 생성 자동화

## 세부 내용 
### 현업에서의 한계/문제점
- 🎯 사전 조사 결과, 마케터가 마케팅 메시지를 작성하는 데 평균 50분이 소요됨
- 💡 가장 어려운 부분: 적절한 마케팅 타겟 설정
### 프로젝트 목표
- 🎯 고객, 상품, 토픽 카테고리 데이터를 활용하여 새로운 세그먼트 구축
  - 🔄 기존 세그먼트는 연령대와 성별 구분만 가능했으며, 관심사와 소비 패턴을 반영하지 못했음
### 단계 1: 세그먼트 정의
- 🔍 방법론: 그래프 마이닝의 "Community Detection" 기법 선택
  - 🔗 클러스터링과 달리, 그래프는 고객, 상품, 토픽 등 다양한 노드 유형과 관계를 반영 가능.
  - 그 중 Louvain Method 적용 
    - 큰 네트워크에 적합
    - 가중 그래프에 적용 가능
    - 여러 노드 유형 반영
- 📝 실험 결과 : 📈 모듈러리티 값이 최대가 되는 시점에서 적절한 커뮤니티 형성됨 (8개 세그먼트)
  - 30~40대 남성, 기혼, 전자 제품 관심
  - 20대 남녀, 활발한 소비 활동
  - 40~60대, 보험, 건강식품, 부동산 연금 등 관심
  - 20~40대 여성, 식재료, 주방 청소용품 등 생활용품 관심
  - ....

### 단계 2: LLM 기반 메시지 생성 
- 🔗 Knowledge Graph 구축: 위에서 찾은 커뮤니티(세그먼트) 기반으로 ('값1', '관계', '값2') 형태의 프롬프트 구성
- 🔄 Chain of Thoughts 단계:
  - 1. 세그먼트 -> 마케팅 메시지
  - 2. 토픽 + 세그먼트 -> 마케팅 메시지
  - 3. 현업에서 사용되는 마케팅 메시지 전략 고려 -> 마케팅 메시지
    - 문장 당 하나의 정보 포함
    - 문장은 간결하고 짧게
    - 불필요한 수식어 제거
    - ....

## 활용 계획
- Topic 모델 + Generation AI 를 결합한 마케팅 플랫폼 개발
- 실제 마케팅에 적용한 A/ B 테스트 진행





# Related Pages (See details for more information 🍀)

https://www.sktuniv.com/9c5b6d48-36ee-48aa-8be9-f76662cb9db4
https://devocean.sk.com/blog/techBoardDetail.do?ID=165042
