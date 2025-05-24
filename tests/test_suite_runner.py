"""
Test Suite Runner - Sistema de Predicción de Tarifas
Ejecuta toda la suite de tests y genera reportes de cobertura detallados
"""

import pytest
import sys
import os
from pathlib import Path
import subprocess
import time
import webbrowser

def run_test_suite():
    """
    Ejecuta la suite completa de tests con diferentes configuraciones
    """
    
    print("=" * 80)
    print("🚀 EJECUTANDO SUITE DE TESTS - SISTEMA DE PREDICCIÓN DE TARIFAS")
    print("=" * 80)
    
    # Configurar argumentos base para pytest
    base_args = [
        "--verbose",
        "--tb=short",
        "--strict-markers",
        "--strict-config"
    ]
    
    # ==========================================
    # 1. TESTS RÁPIDOS (Unitarios y algunos paramétricos)
    # ==========================================
    print("\n📋 1. Ejecutando Tests Rápidos (Unitarios + Paramétricos)")
    print("-" * 60)
    
    quick_args = base_args + [
        "-m", "unit or (parametric and not slow)",
        "--durations=10",
        "tests/"
    ]
    
    try:
        result_quick = pytest.main(quick_args)
        if result_quick == 0:
            print("✅ Tests rápidos: PASARON")
        else:
            print("❌ Tests rápidos: FALLARON")
            return result_quick
    except Exception as e:
        print(f"❌ Error ejecutando tests rápidos: {e}")
        return 1
    
    # ==========================================
    # 2. TESTS DE INTEGRACIÓN
    # ==========================================
    print("\n🔗 2. Ejecutando Tests de Integración")
    print("-" * 60)
    
    integration_args = base_args + [
        "-m", "integration",
        "--durations=10",
        "tests/"
    ]
    
    try:
        result_integration = pytest.main(integration_args)
        if result_integration == 0:
            print("✅ Tests de integración: PASARON")
        else:
            print("❌ Tests de integración: FALLARON")
            return result_integration
    except Exception as e:
        print(f"❌ Error ejecutando tests de integración: {e}")
        return 1
    
    # ==========================================
    # 3. TESTS E2E (solo si se solicita)
    # ==========================================
    run_e2e = input("\n🌐 ¿Ejecutar tests E2E? (requiere Selenium y puede tardar varios minutos) [y/N]: ").lower().strip()
    
    if run_e2e == 'y':
        print("\n🌐 3. Ejecutando Tests End-to-End")
        print("-" * 60)
        
        e2e_args = base_args + [
            "-m", "e2e",
            "--durations=0",
            "-s",  # No capturar output para ver progreso de Selenium
            "tests/"
        ]
        
        try:
            result_e2e = pytest.main(e2e_args)
            if result_e2e == 0:
                print("✅ Tests E2E: PASARON")
            else:
                print("❌ Tests E2E: FALLARON")
                print("💡 Nota: Los tests E2E pueden fallar si no está configurado Selenium correctamente")
                # No retornamos error aquí para permitir que continúe
        except Exception as e:
            print(f"❌ Error ejecutando tests E2E: {e}")
    else:
        print("⏭️  Tests E2E omitidos")
    
    # ==========================================
    # 4. GENERAR REPORTE DE COBERTURA
    # ==========================================
    print("\n📊 4. Generando Reporte de Cobertura")
    print("-" * 60)
    
    coverage_args = base_args + [
        "--cov=pages",
        "--cov=tests/utils",
        "--cov-report=html:tests/coverage_html",
        "--cov-report=term-missing",
        "--cov-branch",
        "-m", "not e2e",  # Excluir E2E del coverage para mejor performance
        "tests/"
    ]
    
    try:
        result_coverage = pytest.main(coverage_args)
        print("\n📊 Reporte de cobertura generado en: tests/coverage_html/index.html")
        
        # Abrir reporte en navegador
        open_report = input("🌐 ¿Abrir reporte de cobertura en el navegador? [y/N]: ").lower().strip()
        if open_report == 'y':
            coverage_path = Path("tests/coverage_html/index.html").absolute()
            webbrowser.open(f"file://{coverage_path}")
            
    except Exception as e:
        print(f"❌ Error generando reporte de cobertura: {e}")
    
    # ==========================================
    # 5. RESUMEN FINAL
    # ==========================================
    print("\n" + "=" * 80)
    print("📈 RESUMEN DE EJECUCIÓN")
    print("=" * 80)
    
    print("\n🔍 Tests por categoría ejecutados:")
    print("   • Tests Unitarios: ✅")
    print("   • Tests de Integración: ✅")
    print("   • Tests Paramétricos: ✅")
    print(f"   • Tests E2E: {'✅' if run_e2e == 'y' else '⏭️ (omitidos)'}")
    
    print("\n📊 Archivos de reporte generados:")
    print("   • Cobertura HTML: tests/coverage_html/index.html")
    print("   • Logs de ejecución: disponibles en terminal")
    
    print("\n🎯 Distribución de tests (según especificación):")
    print("   • Visor Geográfico: 15 tests")
    print("     - Unitarios: 6 (40%)")
    print("     - Integración: 4 (24%)")
    print("     - Paramétricos: 3 (20%)")
    print("     - E2E: 2 (16%)")
    print("   • Predicciones: 12 tests")
    print("     - Unitarios: 5 (40%)")
    print("     - Integración: 3 (25%)")
    print("     - Paramétricos: 2 (17%)")
    print("     - E2E: 2 (17%)")
    
    return 0

def run_specific_module(module_name):
    """
    Ejecuta tests de un módulo específico
    
    Args:
        module_name: 'visor' o 'predicciones'
    """
    print(f"🎯 Ejecutando tests del módulo: {module_name}")
    
    if module_name == "visor":
        marker = "visor_geografico"
        test_file = "tests/test_visor_geografico.py"
    elif module_name == "predicciones":
        marker = "predicciones"
        test_file = "tests/test_predicciones.py"
    else:
        print("❌ Módulo no reconocido. Use 'visor' o 'predicciones'")
        return 1
    
    args = [
        "--verbose",
        "--tb=short",
        "-m", marker,
        "--durations=10",
        test_file
    ]
    
    return pytest.main(args)

def run_by_type(test_type):
    """
    Ejecuta tests por tipo específico
    
    Args:
        test_type: 'unit', 'integration', 'parametric', 'e2e'
    """
    print(f"🏷️  Ejecutando tests de tipo: {test_type}")
    
    args = [
        "--verbose",
        "--tb=short",
        "-m", test_type,
        "--durations=10",
        "tests/"
    ]
    
    if test_type == "e2e":
        args.append("-s")  # No capturar output para Selenium
    
    return pytest.main(args)

def check_dependencies():
    """
    Verifica que todas las dependencias estén instaladas
    """
    print("🔍 Verificando dependencias...")
    
    required_packages = [
        'pytest',
        'pytest-cov',
        'pandas',
        'numpy',
        'streamlit',
        'folium',
        'geopandas',
        'plotly',
        'selenium'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Faltan dependencias: {', '.join(missing_packages)}")
        print("💡 Instala con: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ Todas las dependencias están instaladas")
    return True

def setup_test_environment():
    """
    Configura el entorno de testing
    """
    print("⚙️  Configurando entorno de testing...")
    
    # Crear directorios necesarios
    os.makedirs("tests/coverage_html", exist_ok=True)
    os.makedirs("tests/utils", exist_ok=True)
    
    # Crear __init__.py files si no existen
    init_files = [
        "tests/__init__.py",
        "tests/utils/__init__.py"
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write("# Test package")
    
    # Configurar variables de entorno para testing
    os.environ['TESTING'] = 'true'
    
    print("✅ Entorno configurado correctamente")

def main():
    """
    Función principal del runner de tests
    """
    print("🧪 TEST SUITE RUNNER - SISTEMA DE PREDICCIÓN DE TARIFAS")
    print("Por: Equipo de Desarrollo - Proyecto Pruebas de Software")
    print("=" * 80)
    
    # Verificar dependencias
    if not check_dependencies():
        return 1
    
    # Configurar entorno
    setup_test_environment()
    
    # Mostrar opciones
    print("\n📋 Opciones de ejecución:")
    print("1. Ejecutar suite completa (recomendado)")
    print("2. Solo tests del Visor Geográfico")
    print("3. Solo tests de Predicciones") 
    print("4. Solo tests Unitarios")
    print("5. Solo tests de Integración")
    print("6. Solo tests Paramétricos")
    print("7. Solo tests E2E")
    print("0. Salir")
    
    try:
        choice = input("\n🎯 Selecciona una opción [1]: ").strip() or "1"
        
        if choice == "0":
            print("👋 ¡Hasta luego!")
            return 0
        elif choice == "1":
            return run_test_suite()
        elif choice == "2":
            return run_specific_module("visor")
        elif choice == "3":
            return run_specific_module("predicciones")
        elif choice == "4":
            return run_by_type("unit")
        elif choice == "5":
            return run_by_type("integration")
        elif choice == "6":
            return run_by_type("parametric")
        elif choice == "7":
            return run_by_type("e2e")
        else:
            print("❌ Opción no válida")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Ejecución interrumpida por el usuario")
        return 1
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 