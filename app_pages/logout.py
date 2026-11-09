import streamlit as st


def logout():
    st.session_state.clear()
    st.rerun()


st.title("Logout")
st.write("Are you sure you want to logout?")

if st.button("Confirm Logout", type="primary"):
    logout()

if st.button("Cancel"):
    st.switch_page("app_pages/dashboard.py")
