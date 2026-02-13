"""
CV content analysis service.
Migrated from working_cv_api.py — analyzes CV text and provides
strengths, weaknesses, feedback, and improvement suggestions.
"""
import re


def analyze_cv_content(cv_content: str, filename: str, file_size: int, file_type: str) -> dict:
    """Analyze CV content and return detailed assessment."""
    content_lower = cv_content.lower()
    word_count = len(cv_content.split()) if cv_content else 0

    analysis = extract_cv_sections(cv_content)
    strengths, weaknesses = analyze_strengths_weaknesses(analysis, cv_content)
    completeness_score = calculate_completeness_score(analysis)
    detailed_feedback = create_detailed_feedback(analysis, strengths, weaknesses)
    overall_status, overall_message, grade = assess_cv_quality(
        analysis, completeness_score, strengths, weaknesses
    )

    stats = {
        "word_count": word_count,
        "file_size_mb": round(file_size / (1024 * 1024), 2) if file_size > 0 else 0,
        "completeness_score": completeness_score,
        "completeness_factors": list(analysis.keys()),
        "technical_skills_count": len(analysis.get("technical_skills", [])),
        "experience_years": analysis.get("total_experience_years", 0),
        "projects_count": len(analysis.get("projects", [])),
        "education_level": analysis.get("education_level", "Không xác định"),
    }

    return {
        "overall_assessment": {
            "status": overall_status,
            "score": completeness_score,
            "message": overall_message,
            "grade": grade,
        },
        "statistics": stats,
        "detailed_feedback": detailed_feedback,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "quick_improvements": generate_prioritized_improvements(weaknesses, analysis),
        "ats_friendly_tips": generate_ats_tips(),
        "next_steps": generate_realistic_next_steps(weaknesses, analysis),
    }


def extract_cv_sections(cv_content: str) -> dict:
    """Extract and analyze CV sections."""
    analysis = {}

    # Contact info
    email_match = re.search(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", cv_content
    )
    phone_match = re.search(r"(0|\+84)?[\d\s-]{9,15}", cv_content)
    linkedin_match = re.search(r"linkedin\.com/in/[\w-]+", cv_content, re.IGNORECASE)

    analysis["contact_info"] = {
        "has_email": bool(email_match),
        "has_phone": bool(phone_match),
        "has_linkedin": bool(linkedin_match),
        "email": email_match.group() if email_match else None,
    }

    # Education
    education_patterns = [
        r"(đại học|university|college|cao đẳng)[^.]*",
        r"(bachelor|cử nhân|thạc sĩ|master|phd|tiến sĩ)[^.]*",
        r"(sp_khoa_cntt|khoa công nghệ thông tin|software engineering)[^.]*",
    ]
    education_found = []
    for pattern in education_patterns:
        matches = re.findall(pattern, cv_content, re.IGNORECASE)
        education_found.extend(matches)

    analysis["education"] = {
        "has_education": len(education_found) > 0,
        "details": education_found[:3],
        "education_level": _determine_education_level(education_found),
    }

    # Work experience
    experience_matches = re.findall(
        r"(20\d{2}|01/\d{4}|\d{1,2}/\d{4})[^.]*", cv_content
    )
    analysis["experience"] = {
        "has_experience": len(experience_matches) > 0,
        "experience_entries": experience_matches[:5],
        "total_experience_years": _estimate_experience_years(cv_content),
    }

    # Technical skills
    technical_keywords = [
        "python", "java", "javascript", "nodejs", "react", "vue", "angular",
        "mongodb", "mysql", "postgresql", "docker", "kubernetes", "git",
        "aws", "azure", "gcp", "ci/cd", "agile", "scrum",
    ]
    found_skills = []
    for skill in technical_keywords:
        if re.search(r"\b" + re.escape(skill) + r"\b", cv_content, re.IGNORECASE):
            found_skills.append(skill)
    analysis["technical_skills"] = found_skills

    # Projects
    project_patterns = [
        r"(project|dự án)[^.]*",
        r"github\.com[^s]*",
        r"portfolio[^.]*",
    ]
    projects = []
    for pattern in project_patterns:
        matches = re.findall(pattern, cv_content, re.IGNORECASE)
        projects.extend(matches)
    analysis["projects"] = projects[:3]

    # Certificates
    cert_patterns = [
        r"(certificate|chứng chỉ|certification)[^.]*",
        r"(toeic|ielts|toefl)[^.]*",
    ]
    certificates = []
    for pattern in cert_patterns:
        matches = re.findall(pattern, cv_content, re.IGNORECASE)
        certificates.extend(matches)
    analysis["certificates"] = certificates

    return analysis


def analyze_strengths_weaknesses(analysis: dict, cv_content: str) -> tuple:
    """Analyze specific strengths and weaknesses."""
    strengths = []
    weaknesses = []

    # Technical skills
    tech_skills = analysis.get("technical_skills", [])
    if len(tech_skills) >= 5:
        strengths.append(
            "✅ **Kỹ năng kỹ thuật đa dạng**: Có {} công nghệ lập trình ({})".format(
                len(tech_skills), ", ".join(tech_skills[:5])
            )
        )
    elif len(tech_skills) >= 3:
        strengths.append(
            "✅ **Kỹ năng kỹ thuật tốt**: Biết {} công nghệ ({})".format(
                len(tech_skills), ", ".join(tech_skills)
            )
        )
    elif len(tech_skills) > 0:
        weaknesses.append(
            "❌ **Kỹ năng kỹ thuật còn ít**: Chỉ mới biết {} công nghệ, cần học thêm nhiều hơn".format(
                len(tech_skills)
            )
        )
    else:
        weaknesses.append(
            "❌ **Thiếu kỹ năng kỹ thuật**: CV không ghi rõ công nghệ lập trình đã biết"
        )

    # Experience
    experience_years = analysis.get("experience", {}).get("total_experience_years", 0)
    if experience_years >= 2:
        strengths.append(
            "✅ **Kinh nghiệm thực tế**: Có {} năm kinh nghiệm làm việc".format(experience_years)
        )
    elif experience_years >= 1:
        strengths.append(
            "✅ **Có kinh nghiệm cơ bản**: {} năm kinh nghiệm là điểm khởi đầu tốt".format(experience_years)
        )
    else:
        weaknesses.append(
            "❌ **Thiếu kinh nghiệm làm việc**: Cần có internships hoặc dự án thực tế"
        )

    # Education
    education = analysis.get("education", {})
    if education.get("education_level") == "Đại học":
        strengths.append("✅ **Tốt nghiệp Đại học**: Bằng cấp được nhà tuyển dụng công nhận")
    elif education.get("has_education"):
        strengths.append("✅ **Có nền tảng học vấn**: Đang theo học hoặc đã tốt nghiệp")
    else:
        weaknesses.append("❌ **Thông tin học vấn chưa rõ**: Cần ghi rõ trường và ngành học")

    # Projects
    projects = analysis.get("projects", [])
    if len(projects) >= 2:
        strengths.append(
            "✅ **Dự án cá nhân phong phú**: Có {} dự án chứng minh kỹ năng thực tế".format(len(projects))
        )
    elif len(projects) >= 1:
        strengths.append("✅ **Có dự án cá nhân**: Tốt cho sinh viên mới ra trường")
    else:
        weaknesses.append(
            "❌ **Thiếu dự án cá nhân**: Cần có portfolio hoặc GitHub để chứng minh kỹ năng"
        )

    # Certificates
    certificates = analysis.get("certificates", [])
    if len(certificates) >= 2:
        strengths.append(
            "✅ **Chứng chỉ ngoại ngữ/chuyên môn**: Có {} chứng chỉ tăng uy tín".format(len(certificates))
        )
    elif len(certificates) >= 1:
        strengths.append("✅ **Có chứng chỉ**: Nỗ lực phát triển bản thân")
    else:
        weaknesses.append("❌ **Thiếu chứng chỉ**: Nên có TOEIC/IELTS hoặc chứng chỉ ngành nghề")

    # Contact info
    contact = analysis.get("contact_info", {})
    missing_contact = []
    if not contact.get("has_email"):
        missing_contact.append("email")
    if not contact.get("has_phone"):
        missing_contact.append("số điện thoại")
    if not contact.get("has_linkedin"):
        missing_contact.append("LinkedIn")

    if missing_contact:
        weaknesses.append(
            "❌ **Thông tin liên hệ chưa đầy đủ**: Thiếu {}".format(", ".join(missing_contact))
        )
    else:
        strengths.append("✅ **Thông tin liên hệ đầy đủ**: Dễ dàng liên lạc với nhà tuyển dụng")

    return strengths, weaknesses


def calculate_completeness_score(analysis: dict) -> int:
    """Calculate CV completeness score (0-100)."""
    score = 0

    if analysis.get("contact_info", {}).get("has_email"):
        score += 10
    if analysis.get("contact_info", {}).get("has_phone"):
        score += 5
    if analysis.get("contact_info", {}).get("has_linkedin"):
        score += 5

    if analysis.get("education", {}).get("has_education"):
        score += 20
    if analysis.get("experience", {}).get("has_experience"):
        score += 25

    tech_skills_count = len(analysis.get("technical_skills", []))
    if tech_skills_count >= 5:
        score += 20
    elif tech_skills_count >= 3:
        score += 15
    elif tech_skills_count >= 1:
        score += 10

    if len(analysis.get("projects", [])) >= 2:
        score += 10
    elif len(analysis.get("projects", [])) >= 1:
        score += 5

    if len(analysis.get("certificates", [])) >= 1:
        score += 5

    return min(score, 100)


def create_detailed_feedback(analysis: dict, strengths: list, weaknesses: list) -> list:
    """Create detailed feedback based on analysis."""
    feedback = []

    tech_skills = analysis.get("technical_skills", [])
    if tech_skills:
        feedback.append({
            "section": "Kỹ năng kỹ thuật",
            "status": "good",
            "icon": "✅",
            "title": f"Có {len(tech_skills)} kỹ năng công nghệ",
            "description": f"Bạn đã thành thạo: {', '.join(tech_skills)}. Đây là điểm mạnh cạnh tranh!",
            "tips": [
                f"Highlight các kỹ năng hot nhất: {', '.join(tech_skills[:3])}",
                "Thêm mức độ thành thạo (Basic/Intermediate/Advanced)",
                "Liệt kê các project đã áp dụng từng technology",
            ]
            if len(tech_skills) >= 3
            else [
                "Nên học thêm các công nghệ hot khác",
                "Thực hành thêm qua các dự án cá nhân",
                "Lấy chứng chỉ để xác nhận kỹ năng",
            ],
        })

    exp_years = analysis.get("experience", {}).get("total_experience_years", 0)
    if exp_years >= 1:
        feedback.append({
            "section": "Kinh nghiệm làm việc",
            "status": "good",
            "icon": "✅",
            "title": f"Có {exp_years} năm kinh nghiệm",
            "description": f"{exp_years} năm kinh nghiệm là nền tảng vững chắc cho vị trí junior/mid-level.",
            "tips": [
                "Sử dụng con số cụ thể: 'Tăng performance 30%', 'Quản lý 5 người'",
                "Nêu bật technologies đã dùng trong công việc",
                "Mô tả theo công thức STAR (Situation, Task, Action, Result)",
            ],
        })

    if len(tech_skills) < 3:
        feedback.append({
            "section": "Cần cải thiện kỹ năng",
            "status": "missing",
            "icon": "⚠️",
            "title": f"Chỉ có {len(tech_skills)} kỹ năng công nghệ",
            "description": f"Hiện tại bạn chỉ biết: {', '.join(tech_skills) if tech_skills else 'chưa ghi rõ'}. Cần mở rộng để tăng tính cạnh tranh.",
            "action_items": [
                "Học thêm framework phổ biến (React, Vue, Angular)",
                "Làm quen với database (MongoDB, PostgreSQL)",
                "Học cloud basics (AWS, Azure)",
            ],
        })

    if len(analysis.get("projects", [])) == 0:
        feedback.append({
            "section": "Dự án thực tế",
            "status": "suggestion",
            "icon": "💡",
            "title": "Nên có dự án cá nhân",
            "description": "Dự án cá nhân là cách tốt nhất để chứng minh kỹ năng khi còn ít kinh nghiệm.",
            "action_items": [
                "Tạo GitHub portfolio và đẩy code lên",
                "Làm 2-3 projects từ end-to-end",
                "Deploy projects lên Vercel/Netlify/Railway",
                "Viết README chi tiết cho mỗi project",
            ],
        })

    return feedback


def assess_cv_quality(
    analysis: dict, completeness_score: int, strengths: list, weaknesses: list
) -> tuple:
    """Assess overall CV quality. Returns (status, message, grade)."""
    strength_score = len(strengths) * 10
    weakness_penalty = len(weaknesses) * 8
    final_score = min(100, max(0, completeness_score + strength_score - weakness_penalty))

    if final_score >= 85:
        return (
            "excellent",
            "CV của bạn rất tốt! Có nhiều điểm mạnh và ít điểm yếu. Sẵn sàng cho vị trí Mid-level.",
            "A",
        )
    elif final_score >= 70:
        return (
            "good",
            f"CV khá tốt với {len(strengths)} điểm mạnh. Cần cải thiện {len(weaknesses)} điểm yếu để competitive hơn.",
            "B",
        )
    elif final_score >= 55:
        return (
            "fair",
            f"CV cần cải thiện thêm. Có {len(strengths)} điểm mạnh nhưng còn {len(weaknesses)} điểm yếu cần khắc phục.",
            "C",
        )
    else:
        return (
            "poor",
            f"CV cần cải thiện nhiều. Cần tập trung khắc phục {len(weaknesses)} điểm yếu quan trọng.",
            "D",
        )


def generate_prioritized_improvements(weaknesses: list, analysis: dict) -> list:
    """Generate prioritized improvement suggestions based on weaknesses."""
    improvements = []

    for weakness in weaknesses:
        wl = weakness.lower()
        if "kỹ năng kỹ thuật" in wl:
            improvements.append({
                "priority": "high",
                "title": "Học thêm kỹ năng công nghệ",
                "time_estimate": "2-3 tháng",
                "impact": "Rất cao - Tăng 50% cơ hội phỏng vấn",
            })
        elif "kinh nghiệm" in wl:
            improvements.append({
                "priority": "high",
                "title": "Làm internships hoặc dự án freelance",
                "time_estimate": "1-2 tháng",
                "impact": "Cao - Có kinh nghiệm thực tế",
            })
        elif "dự án" in wl:
            improvements.append({
                "priority": "medium",
                "title": "Xây dựng portfolio 2-3 projects",
                "time_estimate": "1 tháng",
                "impact": "Cao - Chứng minh kỹ năng thực tế",
            })
        elif "chứng chỉ" in wl:
            improvements.append({
                "priority": "medium",
                "title": "Lấy chứng chỉ TOEIC/IELTS",
                "time_estimate": "2-3 tháng",
                "impact": "Trung bình - Yêu cầu của nhiều công ty",
            })
        elif "liên hệ" in wl:
            improvements.append({
                "priority": "high",
                "title": "Cập nhật thông tin liên hệ",
                "time_estimate": "5 phút",
                "impact": "Trung bình - Để nhà tuyển dụng liên lạc",
            })

    return improvements[:5]


def generate_ats_tips() -> list:
    """Generate ATS (Applicant Tracking System) friendly tips."""
    return [
        {"tip": "Sử dụng font đơn giản (Arial, Calibri, Times New Roman)", "reason": "ATS dễ đọc các font tiêu chuẩn"},
        {"tip": "Tránh sử dụng bảng, cột, và đồ họa phức tạp", "reason": "ATS có thể không đọc đúng định dạng phức tạp"},
        {"tip": "Sử dụng từ khóa tiêu chuẩn ngành", "reason": "Giúp CV được tìm thấy dễ dàng hơn"},
        {"tip": "Lưu dưới dạng PDF", "reason": "Định dạng ổn định và bảo toàn layout"},
        {"tip": "Đặt tên file rõ ràng (Ten_Ho_Ten_CV.pdf)", "reason": "Chuyên nghiệp và dễ quản lý"},
    ]


def generate_realistic_next_steps(weaknesses: list, analysis: dict) -> list:
    """Generate realistic next steps based on weaknesses."""
    steps = []

    if any("kỹ năng" in w.lower() for w in weaknesses):
        steps.extend([
            "Chọn 2-3 công nghệ hot (React, Node.js, Python) để học sâu",
            "Làm 2 projects hoàn chỉnh với các công nghệ đã chọn",
            "Đẩy code lên GitHub và viết README chi tiết",
        ])

    if any("kinh nghiệm" in w.lower() for w in weaknesses):
        steps.extend([
            "Tìm internships hoặc freelance projects",
            "Tham gia coding contests hoặc hackathons",
            "Làm volunteer projects cho tổ chức",
        ])

    if any("dự án" in w.lower() for w in weaknesses):
        steps.append("Tạo personal website/portfolio để showcase projects")

    if any("chứng chỉ" in w.lower() for w in weaknesses):
        steps.append("Đăng ký kỳ thi TOEIC/IELTS trong 3 tháng tới")

    steps.extend([
        "Network với developers trên LinkedIn/GitHub",
        "Theo dõi job descriptions để biết market demands",
        "Practice phỏng vấn với bạn bè hoặc mentor",
    ])

    return steps[:6]


# --- Private helpers ---

def _determine_education_level(education_found: list) -> str:
    education_text = " ".join(str(e) for e in education_found).lower()
    if any(w in education_text for w in ["tiến sĩ", "phd", "doctor"]):
        return "Tiến sĩ"
    elif any(w in education_text for w in ["thạc sĩ", "master"]):
        return "Thạc sĩ"
    elif any(w in education_text for w in ["đại học", "university", "bachelor", "cử nhân"]):
        return "Đại học"
    elif any(w in education_text for w in ["cao đẳng", "college"]):
        return "Cao đẳng"
    return "Không xác định"


def _estimate_experience_years(cv_content: str) -> int:
    year_matches = re.findall(r"20\d{2}", cv_content)
    if len(year_matches) >= 2:
        try:
            years = sorted(int(y) for y in year_matches)
            return min(years[-1] - years[0], 10)
        except Exception:
            return 0
    return 0
